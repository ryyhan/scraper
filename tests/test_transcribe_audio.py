"""
Tests for the local audio transcription endpoint (/transcribe-audio/).

The Parakeet model itself is never loaded here — ``AudioTranscriberService.transcribe``
is patched. Decoding/probing tests use real PyAV on a synthetic WAV.
"""

import asyncio
import io
import math
import struct
import wave
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.models import TaskRecord
from app.services import AudioTranscriberService


def _make_wav(seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    """Generate a mono 16-bit PCM sine-wave WAV (8 kHz mimics telephony)."""
    n = int(seconds * sample_rate)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buf.getvalue()


async def _wait_for_status(db_session, task_id: str, timeout: float = 3.0) -> TaskRecord:
    """Poll the DB until the background task leaves IN_PROGRESS."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        db_session.expire_all()
        task = db_session.get(TaskRecord, task_id)
        if task and task.status != "IN_PROGRESS":
            return task
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"Task {task_id} still IN_PROGRESS after {timeout}s")
        await asyncio.sleep(0.05)


# ── Validation ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcribe_rejects_unsupported_extension(client):
    files = {"file": ("notes.txt", b"plain text", "text/plain")}
    response = await client.post("/transcribe-audio/", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_file(client):
    files = {"file": ("call.wav", b"", "audio/wav")}
    response = await client.post("/transcribe-audio/", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_transcribe_rejects_oversize_file(client, monkeypatch):
    monkeypatch.setattr(settings, "AUDIO_MAX_FILE_SIZE_MB", 1)
    files = {"file": ("call.wav", b"\0" * (1024 * 1024 + 1), "audio/wav")}
    response = await client.post("/transcribe-audio/", files=files)
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_transcribe_rejects_non_audio_content(client):
    files = {"file": ("call.mp3", b"definitely not audio data" * 10, "audio/mpeg")}
    response = await client.post("/transcribe-audio/", files=files)
    assert response.status_code == 400


# ── Decoding (real PyAV, no model) ───────────────────────────────────────────

def test_decode_resamples_telephony_wav_to_16k_mono():
    audio = AudioTranscriberService.decode(_make_wav(seconds=2.0, sample_rate=8000))
    assert audio.ndim == 1
    assert str(audio.dtype) == "float32"
    # 2 s at 16 kHz ≈ 32 000 samples (allow resampler edge slack)
    assert abs(len(audio) - 32_000) < 500


# ── Task lifecycle ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcribe_success_lifecycle(client, db_session):
    with patch.object(
        AudioTranscriberService, "transcribe", return_value=("hello world", 1.0)
    ):
        files = {"file": ("call.wav", _make_wav(), "audio/wav")}
        response = await client.post("/transcribe-audio/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "IN_PROGRESS"

        task = await _wait_for_status(db_session, data["task_id"])
        assert task.task_type == "transcribe"
        assert task.status == "SUCCESS"

    poll = await client.get(f"/transcribe-audio/{data['task_id']}")
    assert poll.status_code == 200
    result = poll.json()["result"]
    assert result["text"] == "hello world"
    assert result["filename"] == "call.wav"
    assert result["model"] == settings.TRANSCRIBE_MODEL


@pytest.mark.asyncio
async def test_transcribe_failure_listed_in_failed_endpoint(client, db_session):
    with patch.object(
        AudioTranscriberService, "transcribe", side_effect=RuntimeError("model exploded")
    ):
        files = {"file": ("voicenote.wav", _make_wav(), "audio/wav")}
        response = await client.post("/transcribe-audio/", files=files)
        assert response.status_code == 200
        task = await _wait_for_status(db_session, response.json()["task_id"])
        assert task.status == "FAILURE"
        assert "model exploded" in task.message

    failed = await client.get("/transcribe-audio/failed/")
    assert failed.status_code == 200
    entries = failed.json()["failed_tasks"]
    assert len(entries) == 1
    assert entries[0]["filename"] == "voicenote.wav"


@pytest.mark.asyncio
async def test_transcribe_status_not_found(client):
    response = await client.get("/transcribe-audio/does-not-exist")
    assert response.status_code == 404
