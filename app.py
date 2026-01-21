from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

# Tiny training data (for hackathon demo)
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

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.json["message"]
    vector = vectorizer.transform([msg])
    prediction = model.predict(vector)[0]

    return jsonify({
        "result": "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
