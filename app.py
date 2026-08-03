import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Base setup for Model Loading
base_dir = os.path.abspath(os.path.dirname(__file__))
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

# HTML Layout with Glassmorphism Theme & Animations Embedded Directly
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHT-CET College Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(23, 32, 54, 0.65);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-glow: rgba(99, 102, 241, 0.35);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 20% 20%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 80% 80%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 15px;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 28px;
            padding: 40px;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 36px;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px 16px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.15);
            background: rgba(15, 23, 42, 0.9);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 12px;
            padding: 16px;
            background: var(--primary-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px var(--accent-glow);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px var(--accent-glow);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 32px;
            padding: 24px;
            background: rgba(129, 140, 248, 0.1);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 16px;
            text-align: center;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.98); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-box h3 {
            color: #e0e7ff;
            font-size: 1.15rem;
            font-weight: 600;
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>MHT-CET CAP Predictor</h1>
            <p>Enter details below to run model evaluation</p>
        </header>

        <form action="/predict" method="POST" class="grid-form">
            <div class="input-group">
                <label>Merit Number</label>
                <input type="number" step="any" name="merit_number" required placeholder="e.g. 12500">
            </div>
            <div class="input-group">
                <label>MHTCET Percentile</label>
                <input type="number" step="any" name="mhtcet_percentile" required placeholder="e.g. 97.85">
            </div>
            <div class="input-group">
                <label>Name Code</label>
                <input type="number" step="any" name="name" required placeholder="Encoded Value">
            </div>
            <div class="input-group">
                <label>Gender Code</label>
                <input type="number" step="any" name="gender" required placeholder="Encoded Value">
            </div>
            <div class="input-group">
                <label>Category Code</label>
                <input type="number" step="any" name="category" required placeholder="Encoded Value">
            </div>
            <div class="input-group">
                <label>Seat Alloted Code</label>
                <input type="number" step="any" name="seat_alloted" required placeholder="Encoded Value">
            </div>
            <div class="input-group">
                <label>Course Name Code</label>
                <input type="number" step="any" name="course_name" required placeholder="Encoded Value">
            </div>
            <div class="input-group">
                <label>Seat Number Code</label>
                <input type="number" step="any" name="seat_number" required placeholder="Encoded Value">
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction_text %}
        <div class="result-box">
            <h3>{{ prediction_text }}</h3>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            prediction_text="Error: Model file 'Clgname_model.pkl' could not be loaded on the server."
        )

    try:
        # Extract features safely from HTML input
        merit_number = float(request.form.get("merit_number", 0))
        percentile = float(request.form.get("mhtcet_percentile", 0))
        name = float(request.form.get("name", 0))
        gender = float(request.form.get("gender", 0))
        category = float(request.form.get("category", 0))
        seat_alloted = float(request.form.get("seat_alloted", 0))
        course_name = float(request.form.get("course_name", 0))
        seat_number = float(request.form.get("seat_number", 0))

        # Array shaping for model input
        features = np.array([[
            merit_number, percentile, name, gender, 
            category, seat_alloted, course_name, seat_number
        ]])
        
        prediction = model.predict(features)[0]

        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=f"Predicted Seat / College: {prediction}"
        )

    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=f"Prediction Error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
