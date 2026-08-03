import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Ensure Flask looks in the correct template directory
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Model loading with error safety
MODEL_PATH = os.path.join(base_dir, "Clgname_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html", 
            prediction_text="Error: Model file could not be loaded on the server."
        )

    try:
        # Extract features safely
        merit_number = float(request.form.get("merit_number", 0))
        percentile = float(request.form.get("mhtcet_percentile", 0))
        name = float(request.form.get("name", 0))
        gender = float(request.form.get("gender", 0))
        category = float(request.form.get("category", 0))
        seat_alloted = float(request.form.get("seat_alloted", 0))
        course_name = float(request.form.get("course_name", 0))
        seat_number = float(request.form.get("seat_number", 0))

        features = np.array([[
            merit_number, percentile, name, gender, 
            category, seat_alloted, course_name, seat_number
        ]])
        
        prediction = model.predict(features)[0]

        return render_template(
            "index.html", 
            prediction_text=f"Predicted College / Seat: {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html", 
            prediction_text=f"Prediction Error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
