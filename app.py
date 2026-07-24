from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import os
import logging
import traceback

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===========================
# API Key
# ===========================
API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=API_KEY)

# ===========================
# Models
# ===========================
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

# ===========================
# Home
# ===========================
@app.route("/")
def home():
    return render_template("index.html")


# ===========================
# Health Check
# ===========================
@app.route("/health")
def health():
    return jsonify({
        "status": "running"
    })


# ===========================
# Test Groq
# ===========================
@app.route("/test_groq")
def test_groq():

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": "Reply only OK"
                }
            ],
            max_tokens=5
        )

        return jsonify({
            "success": True,
            "response": response.choices[0].message.content
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================
# Generate Roadmap
# ===========================
@app.route("/generate_roadmap", methods=["POST"])
def generate_roadmap():

    # Accept both FormData and JSON
    data = request.get_json(silent=True)

    if data:
        interest = data.get("interest")
    else:
        interest = request.form.get("interest")

    if not interest:
        return jsonify({
            "error": "Interest is required."
        }), 400

    prompt = f"""
Create a complete roadmap for learning {interest}.

The roadmap should include:

1. Beginner Stage
2. Intermediate Stage
3. Advanced Stage
4. Projects
5. Resources
6. Books
7. YouTube Channels
8. Practice Platforms
9. Interview Preparation
10. Timeline

Return the answer in Markdown format.
"""

    errors = []

    for model in SUPPORTED_MODELS:

        try:

            logger.info(f"Trying model: {model}")

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career mentor."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1200
            )

            roadmap = completion.choices[0].message.content

            return jsonify({
                "success": True,
                "model": model,
                "roadmap": roadmap
            })

        except Exception as e:

            logger.error(f"{model} failed")

            logger.error(str(e))

            logger.error(traceback.format_exc())

            errors.append({
                "model": model,
                "error": str(e)
            })

    return jsonify({
        "success": False,
        "message": "All models failed.",
        "errors": errors
    }), 500


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    app.run(debug=True)