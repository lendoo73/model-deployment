from flask import Flask, render_template, request, jsonify
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email = [request.form.get("content")]
    tokenized_email = tokenizer.transform(email) # this is the preprocessed text
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email_text=email[0])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    email = data["content"]
    if type(email) != list:
        email = [email]
    tokenized_email = tokenizer.transform(email)  # this is the preprocessed text
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

