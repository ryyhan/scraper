"""
Local Audio Transcription Service
---------------------------------
Transcribes audio files fully on-premise using NVIDIA Parakeet TDT 0.6B v3
(multilingual, 25 European languages, automatic language detection) via the
``onnx-asr`` runtime. No audio ever leaves the host.

Pipeline
~~~~~~~~
1. **Decode** — ``PyAV`` (bundles its own FFmpeg libraries, so no system
   ffmpeg is required) decodes any common container/codec (wav, mp3, m4a,
   ogg/opus, flac, webm, amr, …) and resamples to 16 kHz mono float32, which
   is the model's native input format. Telephony audio (8 kHz) is upsampled.
2. **Segment** — Silero VAD splits the waveform into speech segments.
   This is mandatory, not an optimisation: Parakeet can only process roughly
   20–30 s of audio per pass. VAD also drops silence / hold music.
3. **Recognise** — each segment is transcribed by Parakeet (int8-quantised
   ONNX on CPU by default) and the segment texts are joined into plain text.

Model lifecycle
~~~~~~~~~~~~~~~
The model is loaded **once per process** (thread-safe lazy singleton) and
optionally pre-warmed at application startup — see ``main.py``. Loading costs
several seconds and ~1–2 GB RAM, so it must never happen per request.

Heavy imports (``numpy``, ``av``, ``onnx_asr``, ``onnxruntime``) are deferred
to call time so importing this module stays cheap.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from app.core.config import settings

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Native sample rate of Parakeet (and Silero VAD).
TARGET_SAMPLE_RATE = 16_000

#: Accepted upload extensions. Content is additionally verified by PyAV.
ALLOWED_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus",
    ".flac", ".webm", ".amr", ".3gp", ".wma",
})

#: Values of TRANSCRIBE_QUANTIZATION that mean "use the full-precision model".
_NO_QUANTIZATION = {"none", "fp32", "full"}


#: Repository root (…/app/services/audio_transcriber.py → three levels up).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_models_dir() -> Path:
    """
    Absolute model directory. Relative values resolve against the project
    root, not the process CWD, so it works however uvicorn/systemd is launched.
    """
    path = Path(settings.TRANSCRIBE_MODELS_DIR).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _resolve_quantization() -> Optional[str]:
    value = (settings.TRANSCRIBE_QUANTIZATION or "").strip().lower()
    return None if not value or value in _NO_QUANTIZATION else value


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AudioTranscriberService:
    """
    Local speech-to-text using Parakeet TDT 0.6B v3 + Silero VAD.

    Usage::

        AudioTranscriberService.validate(audio_bytes, "call.mp3")
        AudioTranscriberService.probe(audio_bytes)
        text, duration_s = AudioTranscriberService().transcribe(audio_bytes, "call.mp3")
    """

    _model: Any = None
    _load_lock = threading.Lock()

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def max_bytes() -> int:
        return settings.AUDIO_MAX_FILE_SIZE_MB * 1024 * 1024

    @staticmethod
    def validate(audio_bytes: bytes, filename: str) -> None:
        """
        Raise ``ValueError`` if the upload is unacceptable.

        Checks: allowed extension, non-empty, size within
        ``AUDIO_MAX_FILE_SIZE_MB``. The message for an oversize file contains
        "too large" so the route can map it to HTTP 413.
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file type: '{filename}'. Supported audio formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        if not audio_bytes:
            raise ValueError("The uploaded file is empty.")
        if len(audio_bytes) > AudioTranscriberService.max_bytes():
            raise ValueError(
                f"File too large: {len(audio_bytes) / 1024 / 1024:.1f} MB. "
                f"Maximum allowed size is {settings.AUDIO_MAX_FILE_SIZE_MB} MB."
            )

    @staticmethod
    def probe(audio_bytes: bytes) -> None:
        """
        Cheap content check: confirm PyAV can open the container and that it
        has an audio stream. Reads headers only — does not decode the file.
        Raises ``ValueError`` otherwise.
        """
        import av

        try:
            with av.open(io.BytesIO(audio_bytes), mode="r") as container:
                has_audio = bool(container.streams.audio)
        except Exception as exc:
            raise ValueError(
                f"The uploaded file could not be read as audio: {exc}"
            ) from exc
        if not has_audio:
            raise ValueError("The uploaded file contains no audio stream.")

    # ── Decoding ───────────────────────────────────────────────────────────

    @staticmethod
    def decode(audio_bytes: bytes) -> "np.ndarray":
        """
        Decode any supported audio to a 1-D float32 waveform at 16 kHz mono.
        Raises ``ValueError`` if the audio cannot be decoded.
        """
        import av
        import numpy as np

        chunks: list[np.ndarray] = []
        try:
            with av.open(io.BytesIO(audio_bytes), mode="r") as container:
                if not container.streams.audio:
                    raise ValueError("no audio stream found")
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(
                    format="flt", layout="mono", rate=TARGET_SAMPLE_RATE
                )
                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        chunks.append(out.to_ndarray().reshape(-1))
                # Flush samples buffered inside the resampler.
                for out in resampler.resample(None):
                    chunks.append(out.to_ndarray().reshape(-1))
        except Exception as exc:
            raise ValueError(f"Could not decode audio file: {exc}") from exc

        if not chunks:
            raise ValueError("Could not decode audio file: no audio samples found")

        return np.concatenate(chunks).astype(np.float32, copy=False)

    # ── Model lifecycle ────────────────────────────────────────────────────

    @classmethod
    def load_model(cls) -> Any:
        """Return the process-wide VAD-wrapped model, loading it on first use."""
        if cls._model is not None:
            return cls._model
        with cls._load_lock:
            if cls._model is None:
                cls._model = cls._build_model()
        return cls._model

    @staticmethod
    def _build_model() -> Any:
        import onnx_asr
        import onnxruntime as rt

        sess_options = None
        if settings.TRANSCRIBE_NUM_THREADS > 0:
            # Cap CPU usage so transcription does not starve the Playwright
            # browsers sharing this host. 0 = onnxruntime default (all cores).
            sess_options = rt.SessionOptions()
            sess_options.intra_op_num_threads = settings.TRANSCRIBE_NUM_THREADS
            sess_options.inter_op_num_threads = 1

        # First load downloads the weights from Hugging Face into these folders;
        # every later load reads them locally with no network access.
        models_dir = _resolve_models_dir()
        asr_path = str(models_dir / "asr")
        vad_path = str(models_dir / "vad")
        quantization = _resolve_quantization()
        providers = ["CPUExecutionProvider"]

        logger.info(
            f"[Transcriber] Loading model={settings.TRANSCRIBE_MODEL!r} "
            f"quantization={quantization!r} path={asr_path!r} "
            f"threads={settings.TRANSCRIBE_NUM_THREADS or 'default'}"
        )
        t0 = time.perf_counter()

        asr = onnx_asr.load_model(
            settings.TRANSCRIBE_MODEL,
            asr_path,
            quantization=quantization,
            sess_options=sess_options,
            providers=providers,
        )
        vad = onnx_asr.load_vad(
            "silero",
            vad_path,
            sess_options=sess_options,
            providers=providers,
        )
        model = asr.with_vad(
            vad,
            # Keep every segment safely inside Parakeet's ~20–30 s window.
            max_speech_duration_s=settings.TRANSCRIBE_MAX_SEGMENT_SECONDS,
        )

        logger.info(
            f"[Transcriber] Model loaded in {time.perf_counter() - t0:.1f}s"
        )
        return model

    # ── Public API ─────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes, filename: str) -> tuple[str, float]:
        """
        Transcribe *audio_bytes* to plain text. Blocking and CPU-bound —
        call via ``asyncio.to_thread``.

        Returns:
            ``(text, duration_seconds)``. ``text`` is ``""`` when no speech
            was detected.
        """
        t0 = time.perf_counter()
        waveform = self.decode(audio_bytes)
        duration = round(len(waveform) / TARGET_SAMPLE_RATE, 2)
        logger.info(
            f"[Transcriber] Decoded {filename!r}: {duration}s of audio "
            f"in {time.perf_counter() - t0:.2f}s"
        )

        model = self.load_model()
        segments = model.recognize(waveform, sample_rate=TARGET_SAMPLE_RATE)

        texts = [
            seg.text.strip()
            for seg in segments
            if getattr(seg, "text", None) and seg.text.strip()
        ]
        text = " ".join(texts)

        logger.info(
            f"[Transcriber] Transcribed {filename!r}: {len(texts)} segment(s), "
            f"{len(text)} chars, total {time.perf_counter() - t0:.2f}s"
        )
        return text, duration
