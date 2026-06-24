from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    score = None

    if request.method == "POST":

        message = request.form["message"]

        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)

        probability = model.predict_proba(message_vector)

        if prediction[0] == 1:
            result = "🚨 Spam Mail"
            score = round(probability[0][1]*100,2)

        else:
            result = "✅ Ham Mail"
            score = round(probability[0][0]*100,2)

    return render_template(
        "index.html",
        result=result,
        score=score
    )


if __name__ == "__main__":
    app.run(debug=True)
