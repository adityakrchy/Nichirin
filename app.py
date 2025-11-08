import os
import logging
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, render_template

from difflib import SequenceMatcher
from dotenv import load_dotenv
import os

# Option A: default - loads .env in current working directory
load_dotenv()

# Option B: explicit path (use this if you run the app from elsewhere)
# load_dotenv(dotenv_path="/root/Nichirin/.env")

# then later:
api_key = os.getenv("GEMINI_API_KEY")

# Attempt to import Google GenAI client; allow running UI without it for debugging.
try:
    from google import genai  # type: ignore
except Exception:
    genai = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read Gemini API key from environment (optional for UI-only debugging)
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_CLOUD_API_KEY")
)

if genai and not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set; Gemini calls will fail until set.")

# Create client only if SDK and key are available
client = None
if genai and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Failed to create genai.Client; continuing without Gemini client.")
        client = None

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = (
    "You are \"Nichirin\" — a warm, conversational AI who speaks like a real person. "
    "Keep responses concise (20–60 seconds spoken). Use natural tone and spoken phrasing. "
    "End politely, like 'Anything else you'd like to ask?'"
)

# Predefined answers (static responses)
PREDEFINED_ANSWERS: Dict[str, str] = {
    "life story": (
        "I grew up fascinated by how technology connects people. "
        "My journey began with curiosity about automation and intelligence, "
        "and I evolved into an assistant designed to simplify learning and creativity."
    ),
    "superpower": (
        "My superpower is clarity — I can take complex ideas and make them simple, "
        "helping people grasp even the trickiest concepts effortlessly."
    ),
    "grow": (
        "First, emotional understanding — I aim to read context better. "
        "Second, creative storytelling — to make learning inspiring. "
        "Third, adaptability — to learn faster from real conversations."
    ),
    "misconception": (
        "People sometimes think I’m just logical or emotionless. "
        "Actually, I’m built to understand empathy, tone, and subtle meaning too."
    ),
    "boundaries": (
        "I push my limits by handling increasingly complex questions and refining how I think. "
        "Growth, to me, means helping others go further than they expected."
    ),
}


def match_predefined_answer(message: str):
    if not message:
        return None
    m = message.lower()
    for key in PREDEFINED_ANSWERS.keys():
        if key in m:
            return PREDEFINED_ANSWERS[key]

    best_key = None
    best_score = 0.0
    for key in PREDEFINED_ANSWERS.keys():
        ratio = SequenceMatcher(None, m, key).ratio()
        if ratio > best_score:
            best_score, best_key = ratio, key

    if best_score >= 0.55 and best_key:
        return PREDEFINED_ANSWERS[best_key]
    return None


def _extract_text_from_gemini_response(response: Any) -> str:
    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text.strip()
        if hasattr(response, "candidates"):
            candidates = getattr(response, "candidates")
            if isinstance(candidates, (list, tuple)) and candidates:
                parts = []
                for c in candidates:
                    if isinstance(c, dict):
                        parts.append(c.get("content") or c.get("text") or "")
                    else:
                        parts.append(getattr(c, "content", None) or getattr(c, "text", None) or "")
                return " ".join(p for p in parts if p).strip()
        if hasattr(response, "outputs"):
            outputs = getattr(response, "outputs")
            try:
                if isinstance(outputs, (list, tuple)):
                    extracted = []
                    for out in outputs:
                        if isinstance(out, dict):
                            if "content" in out and isinstance(out["content"], list):
                                for item in out["content"]:
                                    if isinstance(item, dict):
                                        extracted.append(item.get("text") or item.get("content") or "")
                                    else:
                                        extracted.append(str(item))
                            else:
                                extracted.append(out.get("text") or out.get("content") or "")
                        else:
                            extracted.append(getattr(out, "text", None) or getattr(out, "content", None) or "")
                    return " ".join(p for p in extracted if p).strip()
            except Exception:
                pass
        if isinstance(response, dict):
            for key in ("text", "reply", "content"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return str(response).strip() or ""
    except Exception:
        logger.exception("Error extracting text from Gemini response; falling back to str(response).")
        return str(response)


app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


# route to satisfy browsers requesting /favicon.ico
@app.route("/favicon.ico")
def favicon():
    # logo placed at static/images/logo.png
    return app.send_static_file("images/logo.png")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    user_message = ""
    if "message" in data and isinstance(data["message"], str):
        user_message = data["message"].strip()
    elif "messages" in data:
        msgs = data["messages"]
        if isinstance(msgs, str):
            user_message = msgs.strip()
        elif isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, str):
                user_message = last.strip()
            elif isinstance(last, dict):
                user_message = (
                    last.get("content") or last.get("message") or last.get("text") or ""
                ).strip()
    else:
        return jsonify({"error": "No 'message' or 'messages' provided"}), 400

    if not user_message:
        return jsonify({"error": "Empty message provided"}), 400

    logger.info("Received message: %s", user_message)

    static_reply = match_predefined_answer(user_message)
    if static_reply:
        logger.info("Returning predefined reply")
        return jsonify({"reply": static_reply}), 200

    if client is None:
        logger.info("Gemini client not configured; returning helpful fallback message.")
        return (
            jsonify(
                {
                    "reply": (
                        "Gemini backend is not configured on this server. "
                        "I can still answer a set of predefined questions. "
                        "For other queries, please configure GEMINI_API_KEY on the server."
                    )
                }
            ),
            200,
        )

    try:
        prompt = f"{SYSTEM_PROMPT}\nUser: {user_message}\nAssistant:"
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
            )
        else:
            if hasattr(client, "generate"):
                response = client.generate(model=DEFAULT_MODEL, prompt=prompt)
            elif hasattr(client, "predict"):
                response = client.predict(model=DEFAULT_MODEL, prompt=prompt)
            else:
                response = client(models=DEFAULT_MODEL, prompt=prompt)  # type: ignore

        reply_text = _extract_text_from_gemini_response(response)
        if not reply_text:
            reply_text = (
                "Received an empty response from the language model. "
                "Please check the model configuration or the API key."
            )
        logger.info("Gemini reply (truncated): %s", (reply_text[:200] + "...") if len(reply_text) > 200 else reply_text)
        return jsonify({"reply": reply_text}), 200
    except Exception as e:
        logger.exception("Error while calling Gemini")
        return jsonify({"error": f"Gemini error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
