"""
PDF Vendor Bill Extractor — Local LLM API
==========================================
Uses Ollama (local LLM) to intelligently extract vendor bill fields
from any PDF format, just like Gemini does but 100% offline.

Engine priority:
  1. Ollama (llama3.2:3b by default) — LLM-powered, handles ANY layout
  2. Regex heuristics                — fast fallback if Ollama is down

Endpoints:
  POST /extract  — Accept PDF as multipart/form-data OR base64 JSON
  GET  /health   — Show status of all engines

Run:
  python3 app.py
  (Requires: pip3 install flask pdfplumber PyMuPDF pytesseract Pillow requests)
  (Requires: ollama serve &&  ollama pull llama3.2:3b)
"""

import io
import re
import json
import base64
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify

# ── PDF text extraction ────────────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz          # PyMuPDF
    import pytesseract
    from PIL import Image
    # Explicitly set tesseract binary path for macOS Homebrew installs
    import shutil
    _tess = (
        shutil.which('tesseract')           # on PATH
        or '/opt/homebrew/bin/tesseract'    # Apple Silicon (M1/M2/M3)
        or '/usr/local/bin/tesseract'       # Intel Mac
    )
    if _tess:
        pytesseract.pytesseract.tesseract_cmd = _tess
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_HOST   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3.2:3b"   # Change to "mistral", "qwen2.5", "phi3", etc.
OLLAMA_TIMEOUT = 120             # seconds

# Fields required by the NetSuite Vendor Bill Suitelet
REQUIRED_FIELDS = {
    "entity":     "The full name of the vendor / company who sent the invoice",
    "subsidiary": "The company or entity being billed (the 'Bill To' party)",
    "tranid":     "The invoice number or reference number",
    "total":      "The grand total / amount due as a numeric value (no symbols)",
    "trandate":   "The invoice issue date in YYYY-MM-DD format",
    "duedate":    "The payment due date in YYYY-MM-DD format",
    "memo":       "A short description of the goods or services provided",
}

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PDF → Text
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_text_pdfplumber(pdf_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def pdf_to_text_ocr(pdf_bytes: bytes) -> str:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        mat  = fitz.Matrix(300 / 72, 300 / 72)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(pages)


def get_text(pdf_bytes: bytes) -> str:
    text = ""
    if HAS_PDFPLUMBER:
        try:
            text = pdf_to_text_pdfplumber(pdf_bytes)
        except Exception as e:
            log.warning(f"pdfplumber failed: {e}")

    if not text.strip() and HAS_OCR:
        log.info("No text found — switching to OCR")
        try:
            text = pdf_to_text_ocr(pdf_bytes)
        except Exception as e:
            log.warning(f"OCR failed: {e}")

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Engine 1 — Ollama Local LLM
# ══════════════════════════════════════════════════════════════════════════════

FIELDS_PROMPT = "\n".join(
    f'  "{k}": {v}' for k, v in REQUIRED_FIELDS.items()
)

SYSTEM_PROMPT = f"""You are an expert invoice data extraction assistant for NetSuite accounting software.
Your job is to read raw invoice text and extract specific fields.
You MUST return ONLY a valid JSON object with exactly these keys:

{FIELDS_PROMPT}

Rules:
- If a field cannot be found, return an empty string "" for text fields or 0 for total.
- Dates MUST be in YYYY-MM-DD format. Convert any other format.
- total MUST be a number (e.g. 1234.56), not a string. No $ or commas.
- entity is the SELLER/VENDOR (who is being paid).
- subsidiary is the BUYER (who is paying — the "Bill To" business name).
- Return ONLY the JSON object. No explanation, no markdown, no code fences.
"""

def ollama_is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def extract_with_ollama(raw_text: str) -> dict:
    """Send raw invoice text to the local Ollama LLM and parse JSON response."""

    # Limit text to ~6000 chars to stay within context window for small models
    truncated_text = raw_text[:6000]

    user_message = f"""Extract the vendor bill fields from this invoice text:

--- INVOICE TEXT START ---
{truncated_text}
--- INVOICE TEXT END ---

Return ONLY the JSON object with the required fields."""

    payload = {
        "model":  OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_message,
        "stream": False,
        "options": {
            "temperature": 0.1,   # near-zero: deterministic, factual
            "top_p": 0.9,
        }
    }

    log.info(f"Calling Ollama model '{OLLAMA_MODEL}' ...")
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT
    )
    response.raise_for_status()

    ai_text = response.json().get("response", "").strip()
    log.info(f"Ollama raw response (first 300 chars): {ai_text[:300]}")

    # Strip any accidental markdown code fences
    ai_text = re.sub(r"^```(?:json)?\s*", "", ai_text, flags=re.MULTILINE)
    ai_text = re.sub(r"```\s*$", "", ai_text, flags=re.MULTILINE).strip()

    # Extract the JSON object even if the model added surrounding text
    json_match = re.search(r"\{.*\}", ai_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in Ollama response: {ai_text[:200]}")

    data = json.loads(json_match.group())

    # Ensure all required keys exist
    for key in REQUIRED_FIELDS:
        if key not in data:
            data[key] = 0 if key == "total" else ""

    # Coerce total to float
    try:
        raw_total = str(data.get("total", "0")).replace(",", "").replace("$", "").strip()
        data["total"] = float(re.sub(r"[^\d.]", "", raw_total) or "0")
    except (ValueError, TypeError):
        data["total"] = 0.0

    return data


# ══════════════════════════════════════════════════════════════════════════════
# Engine 2 — Regex Heuristics (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _first(patterns, text, group=1):
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return m.group(group).strip()
            except IndexError:
                pass
    return ""


def _parse_date(raw):
    if not raw:
        return ""
    raw = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw).strip()
    for fmt in ["%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y",
                "%m/%d/%y", "%d/%m/%y"]:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw


def _parse_amount(raw):
    if not raw:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def extract_with_regex(text: str) -> dict:
    """Heuristic regex extraction — fast but layout-dependent."""
    return {
        "entity": _first([
            r"(?:from|vendor|billed?\s*by|issued?\s*by)[:\s]+([A-Z][^\n]{2,60})",
            r"^([A-Z][A-Z\s,\.&]+(?:Inc|LLC|Ltd|Corp|Co|GmbH|Solutions|Services|Group)[^\n]*)",
        ], text),
        "subsidiary": _first([
            r"(?:bill\s*to|billed?\s*to|client|customer)[:\s]+([A-Z][^\n]{2,80})",
            r"(?:^to)[:\s]+([A-Z][A-Za-z\s,\.]{4,60})",
        ], text),
        "tranid": _first([
            r"(?:invoice\s*(?:no|num|number|#)|inv\s*(?:no|#|num))[\s:#]*([A-Z0-9\-\/]{3,30})",
            r"(?:reference|ref)\s*(?:no|#|num)[\s:#]*([A-Z0-9\-\/]{3,30})",
            r"#\s*([A-Z0-9\-]{4,20})",
        ], text),
        "total": _parse_amount(_first([
            r"(?:total\s+amount\s+due|amount\s+due|grand\s+total|total\s+due|total)[:\s\$]*([0-9,]+\.[0-9]{2})",
            r"(?:balance\s+due|balance)[:\s\$]*([0-9,]+\.[0-9]{2})",
            r"(?:\$|USD|CAD|EUR)\s*([0-9,]+\.[0-9]{2})",
        ], text)),
        "trandate": _parse_date(_first([
            r"(?:invoice\s+date|date\s+of\s+invoice|issued?)[:\s]+([A-Za-z0-9,\s\/\-\.]{5,30})",
            r"date[:\s]+([A-Za-z0-9,\s\/\-\.]{5,20})",
        ], text)),
        "duedate": _parse_date(_first([
            r"(?:due\s+date|payment\s+due|pay\s+by|payable\s+by)[:\s]+([A-Za-z0-9,\s\/\-\.]{5,30})",
            r"due[:\s]+([A-Za-z0-9,\s\/\-\.]{5,20})",
        ], text)),
        "memo": _first([
            r"(?:description|services?\s+description|details|memo)[:\s]+([^\n]{5,200})",
            r"(?:for|regarding|re)[:\s]+([^\n]{5,150})",
        ], text)[:255],
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    ollama_ok   = ollama_is_available()
    model_ready = False
    if ollama_ok:
        try:
            models = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).json()
            model_ready = any(
                m.get("name", "").startswith(OLLAMA_MODEL.split(":")[0])
                for m in models.get("models", [])
            )
        except Exception:
            pass

    return jsonify({
        "status":        "ok",
        "ollama":        ollama_ok,
        "model":         OLLAMA_MODEL,
        "model_ready":   model_ready,
        "pdfplumber":    HAS_PDFPLUMBER,
        "ocr_fallback":  HAS_OCR,
        "regime":        "llm" if ollama_ok else "regex_fallback"
    })


@app.route("/extract", methods=["POST"])
def extract():
    """
    POST /extract

    Option A — multipart form-data:
        curl -X POST http://localhost:5050/extract -F "file=@invoice.pdf"

    Option B — JSON with base64 (matches NetSuite Suitelet format):
        { "base64": "...", "pdf": "...", or "fileBase64": "..." }
    """
    pdf_bytes = None

    # A) File upload
    if "file" in request.files:
        f = request.files["file"]
        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"success": False, "error": "Only PDF files accepted."}), 400
        pdf_bytes = f.read()

    # B) Base64 JSON
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        b64  = body.get("base64") or body.get("pdf") or body.get("fileBase64")
        if b64:
            try:
                pdf_bytes = base64.b64decode(b64)
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64: {e}"}), 400

    if not pdf_bytes:
        return jsonify({
            "success": False,
            "error": "No PDF received. Send 'file' via form-data OR 'base64' via JSON."
        }), 400

    # ── Extract text from PDF ────────────────────────────────────────────
    try:
        raw_text = get_text(pdf_bytes)
    except Exception as e:
        log.exception("Text extraction failed")
        return jsonify({"success": False, "error": f"PDF read error: {e}"}), 500

    if not raw_text.strip():
        if HAS_OCR:
            return jsonify({
                "success": False,
                "error": "OCR ran but extracted no text. The PDF may be blank, corrupted, or in an unsupported image format."
            }), 422
        else:
            return jsonify({
                "success": False,
                "error": "No text found in PDF and OCR is not available. " 
                          "Ensure tesseract is installed: brew install tesseract"
            }), 422

    # ── Try LLM extraction, then regex fallback ──────────────────────────
    engine_used = "regex_fallback"
    data        = {}

    if ollama_is_available():
        try:
            data        = extract_with_ollama(raw_text)
            engine_used = f"ollama/{OLLAMA_MODEL}"
            log.info(f"✅ LLM extraction successful — entity: {data.get('entity')}, total: {data.get('total')}")
        except Exception as e:
            log.warning(f"Ollama extraction failed ({e}), falling back to regex")

    if not data:
        data        = extract_with_regex(raw_text)
        engine_used = "regex_fallback"
        log.info(f"⚠️  Regex extraction — entity: {data.get('entity')}, total: {data.get('total')}")

    return jsonify({
        "success":     True,
        "data":        data,
        "engine_used": engine_used
    })


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PDF Vendor Bill Extractor — Local LLM Edition")
    log.info(f"Ollama model : {OLLAMA_MODEL}")
    log.info(f"pdfplumber   : {'✅' if HAS_PDFPLUMBER else '❌ not installed'}")
    log.info(f"OCR fallback : {'✅' if HAS_OCR else '❌ not installed'}")

    if not ollama_is_available():
        log.warning("⚠️  Ollama is not running. Start it with:  ollama serve")
        log.warning("   Then pull the model with:              ollama pull llama3.2:3b")
    else:
        log.info("Ollama      : ✅ running")

    log.info("=" * 60)
    log.info("API server  : http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
