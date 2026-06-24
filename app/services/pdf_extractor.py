"""
PDF Text Extraction Service
---------------------------
Extracts full text from PDF files using vision-capable LLMs.

Two providers are supported, selectable at call time:

  Gemini (default)
      Uploads the raw PDF bytes via the Gemini Files API. Gemini's native
      PDF understanding handles multi-page documents in a single API call —
      no per-page rendering or splitting required.
      Model: gemini-2.0-flash

  OpenAI
      Renders each PDF page to a JPEG image with ``pypdfium2`` (a pure-Python
      wheel that bundles its own libpdfium binary — no system poppler or
      ghostscript required), then sends all images in a single ``gpt-4o``
      vision call. Falls back to ``gpt-4o-mini`` when the document has more
      than 10 pages to contain token costs.

Design principles
~~~~~~~~~~~~~~~~~
* File-size and MIME-type validation happen before any LLM call so that bad
  input is rejected cheaply and with a clear error message.
* ``tenacity`` retry logic with exponential backoff is applied to every
  network call to handle transient rate-limit and server errors gracefully.
* All side-effects (file upload, temp bytes) are cleaned up or kept in-process
  so the endpoint leaves no artefacts on disk.
* ``loguru`` is used for structured, level-appropriate logging throughout.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Literal

from loguru import logger

from app.services._retry import retry_gemini as _retry_gemini, retry_openai as _retry_openai

from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL = settings.GEMINI_OCR_MODEL
_OPENAI_MODEL_SHORT = settings.OPENAI_OCR_MODEL   # ≤ 10 pages
_OPENAI_MODEL_LONG = settings.OPENAI_OCR_MODEL    # > 10 pages (same model, preserved for easy swap)
_MAX_PAGES_FULL_MODEL = 10            # threshold above which we switch to the cheaper model

_RENDER_DPI = 150        # 150 DPI balances OCR quality vs. payload size
_JPEG_QUALITY = 85       # JPEG quality for OpenAI image encoding

_EXTRACTION_PROMPT = """\
You are an expert document reader specialised in processing background-check \
and HR screening forms, including scanned and handwritten documents.

Your task is to extract the COMPLETE content of this document and render it \
as plain text that perfectly represents what a human reader would understand \
from the page — including all visual selection cues.

GENERAL RULES
─────────────
• Preserve natural reading order and page structure.
• Separate pages with the exact marker:  --- Page N ---  (N = page number).
• Do NOT summarise, paraphrase, or omit any content.
• Return ONLY the extracted text — no commentary, no markdown fences.

VISUAL SELECTION CUES (critical for forms and checkboxes)
─────────────────────────────────────────────────────────
Many fields in these documents are answered by marking a checkbox, bubble, or
option rather than writing text. You MUST detect and faithfully transcribe
every such mark using the notation below.

Detection guidance — treat the following as a SELECTED mark:
  • A tick / check-mark  (✓, ✔, or any hand-drawn check)
  • An X or cross        (✗, ×, or any hand-drawn X)
  • A filled or shaded circle / bubble
  • A circled option     (when a word or box is ringed/circled by hand)
  • A scribble, shade, or heavy pen stroke inside or over a box/option
  • An underline beneath a specific option (when used as selection)

Treat the following as NOT SELECTED:
  • An empty box  □  or empty circle  ○
  • An option with no mark near it
  • A printed horizontal line used as a blank fill-in field (e.g. "Average hrs/wk: ______")
    — these are writing blanks, NOT selection marks, even if they appear next to an option.
  • A pre-printed dash or underscore that is part of the form's layout/template

IMPORTANT — "underline as selection" rule:
  Only treat an underline as [SELECTED] if it is a hand-drawn mark placed directly
  beneath a specific word or label to indicate choice. Do NOT apply this to:
  • Printed fill-in lines (longer horizontal rules used as writing spaces)
  • Lines that appear consistently next to every option in a list (layout artifact)
  • Dashes or underscores that are part of the form's pre-printed template

Transcription notation — inline with the surrounding text:
  • For a checkbox:
      Marked   → append  [SELECTED]   immediately after the option label
      Unmarked → append  [NOT SELECTED]
  • For a multiple-choice row (where one of several options is circled/ticked):
      Write all options in order; append [SELECTED] to the chosen one only.

EXAMPLES
────────
Checkbox:
  [ ] Full-time  [✓] Part-time  [ ] Contract
  →  Full-time [NOT SELECTED]  Part-time [SELECTED]  Contract [NOT SELECTED]

Multiple-choice (circled answer):
  Employment Status:  Employed  Self-employed  Unemployed  (where "Employed" is circled)
  →  Employment Status: Employed [SELECTED]  Self-employed [NOT SELECTED]  Unemployed [NOT SELECTED]

Filled bubble (scantron-style):
  ● Yes  ○ No
  →  Yes [SELECTED]  No [NOT SELECTED]

If a box or option is ambiguous (e.g., a very faint mark or a stray pen stroke
that does not clearly indicate intent), append  [UNCLEAR]  instead of guessing.

Now extract the full document text following all rules above.\
"""

# ---------------------------------------------------------------------------
# Retry helpers — imported from the shared module (app/services/_retry.py)
# ---------------------------------------------------------------------------
# _retry_gemini and _retry_openai are imported at the top of this file.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_repetition_loop(text: str) -> bool:
    """
    Detect whether Gemini has entered a token-repetition loop.

    Strategy: take the first 200 characters of the output as a fingerprint
    and count how many times it occurs in the full text.  A legitimate
    extraction — even a 30-page dense document — will never repeat the same
    200-char block 10+ times.  A repetition loop always will.

    The length guard ensures short documents (< 2 000 chars) are never flagged.
    """
    _SAMPLE_SIZE = 200
    _REPEAT_THRESHOLD = 10

    if len(text) < _SAMPLE_SIZE * _REPEAT_THRESHOLD:
        return False

    sample = text[:_SAMPLE_SIZE]
    return text.count(sample) >= _REPEAT_THRESHOLD


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PdfExtractorService:
    """
    Extracts plain text from a PDF file using either Gemini or OpenAI vision.

    Usage::

        service = PdfExtractorService()
        text, page_count = service.extract(pdf_bytes, "invoice.pdf", "gemini")
    """

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def validate(pdf_bytes: bytes, filename: str) -> None:
        """
        Raise ``ValueError`` if the file is invalid.

        Checks performed:
          1. Extension must be ``.pdf`` (case-insensitive).
          2. File size must not exceed ``settings.PDF_MAX_FILE_SIZE_MB`` MB.
          3. PDF magic bytes (``%PDF``) must be present at the start.
        """
        max_bytes = settings.PDF_MAX_FILE_SIZE_MB * 1024 * 1024

        if not filename.lower().endswith(".pdf"):
            raise ValueError(
                f"Invalid file type: '{filename}'. Only PDF files are accepted."
            )

        if len(pdf_bytes) > max_bytes:
            raise ValueError(
                f"File too large: {len(pdf_bytes) / 1024 / 1024:.1f} MB. "
                f"Maximum allowed size is {settings.PDF_MAX_FILE_SIZE_MB} MB."
            )

        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(
                "The uploaded file does not appear to be a valid PDF "
                "(missing PDF magic bytes)."
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(
        self,
        pdf_bytes: bytes,
        filename: str,
        provider: Literal["gemini", "openai"],
    ) -> tuple[str, int]:
        """
        Extract text from *pdf_bytes* using the specified *provider*.

        Args:
            pdf_bytes: Raw PDF file content.
            filename:  Original filename (used for logging and validation).
            provider:  ``"gemini"`` or ``"openai"``.

        Returns:
            A tuple of ``(extracted_text, page_count)``.

        Raises:
            ValueError:   On validation failure (bad extension, size, magic bytes).
            RuntimeError: If the required API key is not configured.
        """
        self.validate(pdf_bytes, filename)

        logger.info(
            f"[PdfExtractor] Starting extraction: file={filename!r}, "
            f"size={len(pdf_bytes) / 1024:.1f} KB, provider={provider!r}"
        )

        if provider == "gemini":
            return self._extract_gemini(pdf_bytes, filename)
        else:
            return self._extract_openai(pdf_bytes, filename)

    # ── Gemini path ─────────────────────────────────────────────────────────

    def _require_gemini_client(self):
        """Return a configured ``genai.Client`` or raise clearly."""
        from google import genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            )
        # Dual-layer retry (inner): SDK retries 2x reading Retry-After headers
        # before tenacity's outer-layer takes over.  timeout is in MILLISECONDS.
        from google.genai import types as _genai_types
        return genai.Client(
            api_key=api_key,
            http_options=_genai_types.HttpOptions(
                timeout=180_000,
                retry_options=_genai_types.HttpRetryOptions(attempts=2),
            ),
        )

    def _extract_gemini(self, pdf_bytes: bytes, filename: str) -> tuple[str, int]:
        """
        Upload PDF bytes to the Gemini Files API and run a single extraction call.

        Gemini's native PDF support handles the entire document — no page
        splitting or rendering is needed. The Files API accepts the raw bytes
        directly (no temporary file on disk).
        """
        from google import genai
        from google.genai import types as genai_types

        client = self._require_gemini_client()

        # ── Upload to Files API ──────────────────────────────────────────
        logger.debug("[PdfExtractor][Gemini] Uploading PDF to Files API…")

        @_retry_gemini()
        def _upload() -> "genai.types.File":  # type: ignore[name-defined]
            return client.files.upload(
                file=io.BytesIO(pdf_bytes),
                config=genai_types.UploadFileConfig(
                    mime_type="application/pdf",
                    display_name=filename,
                ),
            )

        uploaded_file = _upload()
        logger.debug(
            f"[PdfExtractor][Gemini] File uploaded: uri={uploaded_file.uri!r}"
        )

        # ── Generate content ─────────────────────────────────────────────
        @_retry_gemini()
        def _generate() -> str:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type="application/pdf",
                    ),
                    _EXTRACTION_PROMPT,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            return response.text or ""

        extracted_text = _generate()

        # ── Clean up the uploaded file ───────────────────────────────────
        try:
            client.files.delete(name=uploaded_file.name)
            logger.debug(
                f"[PdfExtractor][Gemini] Deleted uploaded file: {uploaded_file.name!r}"
            )
        except Exception as cleanup_err:
            # Non-fatal — the Files API automatically purges files after 48 h.
            logger.warning(
                f"[PdfExtractor][Gemini] Could not delete uploaded file "
                f"{uploaded_file.name!r}: {cleanup_err}"
            )

        # ── Repetition-loop detection ────────────────────────────────
        # Checked OUTSIDE the retry decorator: retrying the same PDF with
        # the same prompt will produce the same loop.  Surface a clear error
        # so the caller can decide to switch provider or inspect the file.
        if _is_repetition_loop(extracted_text):
            raise RuntimeError(
                f"[PdfExtractor][Gemini] Repetition loop detected for "
                f"{filename!r} — output was {len(extracted_text):,} chars. "
                f"The PDF may contain content that confuses the vision model. "
                f"Try the OpenAI provider as an alternative."
            )

        # ── Safety truncation ────────────────────────────────────────────
        # Last-resort cap for pathological cases not caught above.
        # 150 000 chars ≈ 37 500 tokens — comfortably covers 30+ dense pages.
        _MAX_CHARS = 150_000
        if len(extracted_text) > _MAX_CHARS:
            logger.warning(
                f"[PdfExtractor][Gemini] Output unusually large "
                f"({len(extracted_text):,} chars) — truncating to {_MAX_CHARS:,}."
            )
            extracted_text = extracted_text[:_MAX_CHARS]

        # Estimate page count from page markers inserted by the prompt.
        page_count = self._count_pages_from_markers(extracted_text)

        logger.info(
            f"[PdfExtractor][Gemini] Extraction complete: "
            f"~{page_count} pages, {len(extracted_text)} chars"
        )
        return extracted_text, page_count

    # ── OpenAI path ─────────────────────────────────────────────────────────

    def _require_openai_client(self):
        """Return a configured ``OpenAI`` client or raise clearly."""
        from openai import OpenAI

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file and restart the server."
            )
        # max_retries=2: SDK inner-layer reads Retry-After headers on 429/5xx
        # before our tenacity outer-layer (in _retry.py) takes over.
        return OpenAI(api_key=api_key, max_retries=2)

    def _render_pages(self, pdf_bytes: bytes) -> list[str]:
        """
        Render every PDF page to a base64-encoded JPEG string.

        Uses ``pypdfium2`` — a zero-system-dependency PDF renderer that ships
        its own libpdfium binary. Renders at ``_RENDER_DPI`` DPI.

        Returns:
            List of base64-encoded JPEG strings, one per page.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "pypdfium2 is required for the OpenAI provider. "
                "Install it with: pip install pypdfium2"
            ) from exc

        pdf = pdfium.PdfDocument(pdf_bytes)
        images_b64: list[str] = []

        scale = _RENDER_DPI / 72.0  # pdfium works in 72 DPI points

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=scale, rotation=0)
            pil_image = bitmap.to_pil()

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            images_b64.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

            logger.debug(
                f"[PdfExtractor][OpenAI] Rendered page {page_index + 1}/{len(pdf)}"
            )

        pdf.close()
        return images_b64

    def _extract_openai(self, pdf_bytes: bytes, filename: str) -> tuple[str, int]:
        """
        Render PDF pages to images and send them to ``gpt-4o`` vision.

        All page images are sent in a single API call to preserve cross-page
        context. The cheaper ``gpt-4o-mini`` model is used when the document
        exceeds ``_MAX_PAGES_FULL_MODEL`` pages.
        """
        client = self._require_openai_client()

        # ── Render pages ─────────────────────────────────────────────────
        logger.debug("[PdfExtractor][OpenAI] Rendering PDF pages…")
        images_b64 = self._render_pages(pdf_bytes)
        page_count = len(images_b64)

        model = (
            _OPENAI_MODEL_SHORT
            if page_count <= _MAX_PAGES_FULL_MODEL
            else _OPENAI_MODEL_LONG
        )
        logger.debug(
            f"[PdfExtractor][OpenAI] {page_count} page(s) rendered, "
            f"using model={model!r}"
        )

        # ── Build vision message content ─────────────────────────────────
        # Each page image is annotated with a page label so the model can
        # reference page numbers in its "--- Page N ---" markers.
        content: list[dict] = [{"type": "text", "text": _EXTRACTION_PROMPT}]
        for i, img_b64 in enumerate(images_b64, start=1):
            content.append({"type": "text", "text": f"[Page {i}]"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high",
                },
            })

        # ── Call the vision model ────────────────────────────────────────
        @_retry_openai()
        def _call_vision() -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=16000,
            )
            return response.choices[0].message.content or ""

        extracted_text = _call_vision()

        logger.info(
            f"[PdfExtractor][OpenAI] Extraction complete: "
            f"{page_count} pages, {len(extracted_text)} chars"
        )
        return extracted_text, page_count

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def _count_pages_from_markers(text: str) -> int:
        """
        Count ``--- Page N ---`` markers inserted by the extraction prompt.

        Returns at least 1 so that an empty response doesn't report 0 pages.
        """
        import re
        matches = re.findall(r"---\s*Page\s+\d+\s*---", text, flags=re.IGNORECASE)
        return max(len(matches), 1)
