import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model file paths
COLLEGE_MODEL_PATH = 'Clgname_model.pkl'
COURSE_MODEL_PATH = 'coursename_model.pkl'

# Load trained models safely
def load_model(filepath):
    if os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    else:
        print(f"Warning: File {filepath} not found.")
        return None

college_model = load_model(COLLEGE_MODEL_PATH)
course_model = load_model(COURSE_MODEL_PATH)

# Single merged HTML Template string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Admission & Course Predictor</title>
    <!-- Google Fonts & Font Awesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-cyan: #06b6d4;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 960px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-wrapper i {
            position: absolute;
            left: 1rem;
            color: var(--accent-cyan);
            font-size: 0.9rem;
        }

        .input-wrapper input {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: #fff;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-wrapper input:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
            background: rgba(15, 23, 42, 0.85);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink));
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: transform 0.2s ease, box-shadow 0.3s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(236, 72, 153, 0.3);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .results-container {
            margin-top: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.5rem;
            opacity: 0;
            transform: translateY(15px);
            transition: all 0.5s ease-out;
        }

        .results-container.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .result-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .result-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
        }

        .result-card.college::before { background: var(--accent-cyan); }
        .result-card.course::before { background: var(--accent-pink); }

        .result-card h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-card .value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff;
            word-break: break-word;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>College & Course Admission Predictor</h1>
        <p>Enter student details below to predict the allocated college and course concurrently.</p>
    </div>

    <form id="predictionForm" class="grid-form">
        <div class="input-group">
            <label for="merit_number">Merit Number</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-list-ol"></i>
                <input type="number" id="merit_number" required placeholder="e.g. 12050">
            </div>
        </div>

        <div class="input-group">
            <label for="mhtcet_percentile">MHTCET Percentile</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-chart-line"></i>
                <input type="number" step="any" id="mhtcet_percentile" required placeholder="e.g. 98.45">
            </div>
        </div>

        <div class="input-group">
            <label for="name_code">Name (Encoded)</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-user"></i>
                <input type="number" id="name_code" required placeholder="e.g. 45">
            </div>
        </div>

        <div class="input-group">
            <label for="gender_code">Gender (Encoded)</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-venus-mars"></i>
                <input type="number" id="gender_code" required placeholder="e.g. 1">
            </div>
        </div>

        <div class="input-group">
            <label for="category_code">Category (Encoded)</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-layer-group"></i>
                <input type="number" id="category_code" required placeholder="e.g. 3">
            </div>
        </div>

        <div class="input-group">
            <label for="seat_alloted_code">Seat Alloted (Encoded)</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-chair"></i>
                <input type="number" id="seat_alloted_code" required placeholder="e.g. 2">
            </div>
        </div>

        <div class="input-group">
            <label for="institute_code">Institute Code</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-building-columns"></i>
                <input type="number" id="institute_code" required placeholder="e.g. 6146">
            </div>
        </div>

        <div class="input-group">
            <label for="seat_number_code">Seat Number Code</label>
            <div class="input-wrapper">
                <i class="fa-solid fa-id-card"></i>
                <input type="number" id="seat_number_code" required placeholder="e.g. 1024">
            </div>
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">
            <span class="spinner" id="btnSpinner"></span>
            <span id="btnText"><i class="fa-solid fa-wand-magic-sparkles"></i> Predict Admission Details</span>
        </button>
    </form>

    <div class="results-container" id="resultsContainer">
        <div class="result-card college">
            <h3>Predicted College Name</h3>
            <div class="value" id="collegeResult">--</div>
        </div>
        <div class="result-card course">
            <h3>Predicted Course Name</h3>
            <div class="value" id="courseResult">--</div>
        </div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const btnSpinner = document.getElementById('btnSpinner');
        const btnText = document.getElementById('btnText');
        const submitBtn = document.getElementById('submitBtn');
        const resultsContainer = document.getElementById('resultsContainer');

        // Show loading state
        btnSpinner.style.display = 'inline-block';
        btnText.textContent = 'Processing...';
        submitBtn.disabled = true;

        const payload = {
            merit_number: document.getElementById('merit_number').value,
            mhtcet_percentile: document.getElementById('mhtcet_percentile').value,
            name_code: document.getElementById('name_code').value,
            gender_code: document.getElementById('gender_code').value,
            category_code: document.getElementById('category_code').value,
            seat_alloted_code: document.getElementById('seat_alloted_code').value,
            institute_code: document.getElementById('institute_code').value,
            seat_number_code: document.getElementById('seat_number_code').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.status === 'success') {
                document.getElementById('collegeResult').innerText = result.predicted_college;
                document.getElementById('courseResult').innerText = result.predicted_course;
                resultsContainer.classList.add('visible');
            } else {
                alert('Prediction failed: ' + result.message);
            }
        } catch (error) {
            alert('An error occurred while calling the prediction endpoint.');
            console.error(error);
        } finally {
            btnSpinner.style.display = 'none';
            btnText.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Admission Details';
            submitBtn.disabled = false;
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    # Render the HTML template directly from the string
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Parse form inputs
        merit_number = float(data.get('merit_number', 0))
        mhtcet_percentile = float(data.get('mhtcet_percentile', 0))
        name_code = float(data.get('name_code', 0))
        gender_code = float(data.get('gender_code', 0))
        category_code = float(data.get('category_code', 0))
        seat_alloted_code = float(data.get('seat_alloted_code', 0))
        institute_code = float(data.get('institute_code', 0))
        seat_number_code = float(data.get('seat_number_code', 0))

        # 8 Input features format:
        # [Merit Number, MHTCET Percentile, Name, Gender, Category, Seat Alloted, Institute Name, Seat Number]
        features = np.array([[
            merit_number,
            mhtcet_percentile,
            name_code,
            gender_code,
            category_code,
            seat_alloted_code,
            institute_code,
            seat_number_code
        ]])

        college_pred = "Model Not Loaded"
        course_pred = "Model Not Loaded"

        if college_model is not None:
            pred_clg = college_model.predict(features)[0]
            college_pred = str(pred_clg)

        if course_model is not None:
            pred_course = course_model.predict(features)[0]
            course_pred = str(pred_course)

        return jsonify({
            'status': 'success',
            'predicted_college': college_pred,
            'predicted_course': course_pred
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
