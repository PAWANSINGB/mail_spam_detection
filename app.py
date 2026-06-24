from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
# Make sure tune nayi files save ki hain jisme n-grams (1,2) use hua tha
model = joblib.load("spam_model.pkl") 
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None

    if request.method == "POST":
        message = request.form["message"]

        # 1. Text ko transform kiya
        message_vector = vectorizer.transform([message])

        # 2. Direct predict() use karne ki jagah SPAM hone ki probability nikali
        # probability[0][1] matlab spam hone ka chance (0 se 1 ke beech)
        spam_prob = model.predict_proba(message_vector)[0][1]

        # 3. Custom Threshold (0.35) check lagaya
        if spam_prob >= 0.35:
            result = "🚨 Spam Mail"
            score = round(spam_prob * 100, 2)  # Kitne % chance hain ki spam hai
        else:
            result = "✅ Ham Mail"
            # Ham ki probability dikhane ke liye 100 mein se spam ki prob minus kar do
            score = round((1 - spam_prob) * 100, 2) 

    return render_template(
        "index.html",
        result=result,
        score=score
    )


if __name__ == "__main__":
    app.run(debug=True)
