from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/")
def home():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    news_text = data.get("text", "")

    if not news_text:
        return jsonify({"error": "Please enter news text."})

    cleaned_text = clean_text(news_text)
    transformed_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(transformed_text)[0]

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
