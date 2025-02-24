create flask app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load the model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Diabetes Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    return jsonify({"Diabetic": int(prediction)})

if _name_ == "_main_":
    app.run(debug=True)