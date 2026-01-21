from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Get API key safely
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("⚠️ API_KEY not found (OK for ML-only app)")
else:
    print("✅ API_KEY loaded successfully")

# -------------------------------
# Tiny training data (Hackathon demo)
# -------------------------------
messages = [
    "Win cash now",
    "Congratulations you won a prize",
    "Free entry offer",
    "Call me when you are free",
    "Let's meet tomorrow",
    "How are you doing"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Not Spam

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

# -------------------------------
# API Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    msg = data.get("message", "")

    vector = vectorizer.transform([msg])
    prediction = model.predict(vector)[0]

    return jsonify({
        "result": "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    })

# -------------------------------
# Run App (Render-ready)
# -------------------------------
@app.route("/")
def home():
    return "AI Spam Detector Backend is Running 🚀"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

