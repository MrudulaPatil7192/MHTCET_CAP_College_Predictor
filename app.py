import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "Clgname_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found on server."}), 500

    try:
        # Extract features from form input
        data = request.form
        features = [
            float(data.get("merit_number", 0)),
            float(data.get("mhtcet_percentile", 0)),
            float(data.get("name", 0)),
            float(data.get("gender", 0)),
            float(data.get("category", 0)),
            float(data.get("seat_alloted", 0)),
            float(data.get("course_name", 0)),
            float(data.get("seat_number", 0))
        ]
        
        # Reshape for prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        return render_template("index.html", prediction_text=f"Predicted Result / Class: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
