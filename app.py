import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model File Path
MODEL_PATH = "Clgname_model_pkl"

# Optional: Map predicted numeric label IDs to actual College Names
# Add/edit your label-encoder mapping here if you have standard encoded classes
COLLEGE_MAP = {
    0: "COEP Technological University, Pune",
    1: "VJTI, Mumbai",
    2: "PICT, Pune",
    3: "VIT, Pune",
    4: "Walchand College of Engineering, Sangli",
    5: "PCCOE, Pune",
    6: "MIT-WPU, Pune"
}

# Load the Scikit-Learn Model
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found!")

# Modern Emerald & Dark Slate Dashboard UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHTCET CAP College Predictor</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans min-h-screen flex flex-col">

    <!-- Header Navbar -->
    <header class="bg-slate-800 border-b border-slate-700 p-4 shadow-md">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-3xl">🎓</span>
                <h1 class="text-xl font-bold tracking-wide text-emerald-400">MHTCET CAP College Predictor</h1>
            </div>
            <span class="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full font-medium">
                Model: Active
            </span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto my-8 px-4 flex-grow w-full">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

            <!-- Input Form Panel -->
            <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl lg:col-span-1">
                <h2 class="text-lg font-semibold text-slate-200 mb-6 border-b border-slate-700 pb-3 flex items-center">
                    <span class="mr-2">📋</span> Candidate Details
                </h2>

                <form id="predictorForm" class="space-y-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Merit Number</label>
                        <input type="number" id="merit_number" value="6008" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">MHTCET Percentile</label>
                        <input type="number" step="0.001" id="mhtcet_percentile" value="75.0" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Candidate Name Code</label>
                        <input type="number" id="name" value="-3" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Gender Code</label>
                            <input type="number" id="gender" value="1" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Category Code</label>
                            <input type="number" id="category" value="-8" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Seat Alloted Code</label>
                            <input type="number" id="seat_alloted" value="-5" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Course Code</label>
                            <input type="number" id="course_name" value="6" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Seat Number Code</label>
                        <input type="number" id="seat_number" value="101" required class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded-lg p-2.5 focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                    </div>

                    <button type="submit" id="submitBtn" class="w-full mt-2 bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-3 px-4 rounded-lg transition duration-200 flex justify-center items-center">
                        <span>Predict Allocation</span>
                    </button>
                </form>
            </div>

            <!-- Visualization / Output Panel -->
            <div class="lg:col-span-2 space-y-6">
                
                <!-- Prediction Result Card -->
                <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-2 h-full bg-emerald-500"></div>
                    <div class="flex justify-between items-start">
                        <div>
                            <span class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Predicted College Result</span>
                            <h3 id="predictedCollege" class="text-2xl font-bold text-slate-100 mt-2">--</h3>
                            <p id="predictedCode" class="text-sm text-slate-400 mt-1">Submit parameters to run inference.</p>
                        </div>
                        <div class="p-3 bg-slate-900/50 rounded-lg border border-slate-700 text-2xl">🏛️</div>
                    </div>
                </div>

                <!-- Probability Chart Card -->
                <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
                    <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center">
                        <span class="mr-2">📊</span> Probability Analytics
                    </h3>
                    <div class="relative h-72 w-full">
                        <canvas id="probabilityChart"></canvas>
                    </div>
                </div>

            </div>

        </div>
    </main>

    <!-- AJAX Script -->
    <script>
        let chartInstance = null;

        document.getElementById('predictorForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const btn = document.getElementById('submitBtn');
            btn.innerText = "Predicting...";
            btn.disabled = true;

            const payload = {
                merit_number: parseFloat(document.getElementById('merit_number').value),
                mhtcet_percentile: parseFloat(document.getElementById('mhtcet_percentile').value),
                name: parseFloat(document.getElementById('name').value),
                gender: parseFloat(document.getElementById('gender').value),
                category: parseFloat(document.getElementById('category').value),
                seat_alloted: parseFloat(document.getElementById('seat_alloted').value),
                course_name: parseFloat(document.getElementById('course_name').value),
                seat_number: parseFloat(document.getElementById('seat_number').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();

                if (result.status === "success") {
                    document.getElementById('predictedCollege').innerText = result.college_name;
                    document.getElementById('predictedCode').innerText = "Model Predicted ID: " + result.prediction;

                    if (result.probabilities && result.probabilities.length > 0) {
                        renderChart(result.probabilities, result.classes);
                    }
                } else {
                    alert("Prediction Error: " + result.message);
                }
            } catch (err) {
                alert("Server Connection Failed: " + err.message);
            } finally {
                btn.innerText = "Predict Allocation";
                btn.disabled = false;
            }
        });

        function renderChart(probabilities, classes) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            const labels = classes ? classes.map(c => `College ${c}`) : probabilities.map((_, i) => `Class ${i}`);

            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Allocation Confidence Probability',
                        data: probabilities,
                        backgroundColor: 'rgba(16, 185, 129, 0.6)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8' } }
                    },
                    scales: {
                        y: { 
                            beginAtZero: true, 
                            max: 1,
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        },
                        x: { 
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"status": "error", "message": "Model 'Clgname_model_pkl' file not found or failed to load on server."}), 500

    try:
        data = request.get_json()

        # Extract features array in exact dataset order
        features = np.array([[
            float(data.get("merit_number", 0)),
            float(data.get("mhtcet_percentile", 0)),
            float(data.get("name", 0)),
            float(data.get("gender", 0)),
            float(data.get("category", 0)),
            float(data.get("seat_alloted", 0)),
            float(data.get("course_name", 0)),
            float(data.get("seat_number", 0))
        ]])

        # Execute Model Prediction
        prediction_val = model.predict(features)[0]
        predicted_id = int(prediction_val)

        # Get mapped college name if defined, else format string
        college_name = COLLEGE_MAP.get(predicted_id, f"Allocated College (ID: {predicted_id})")

        # Extract Probability Distribution
        probabilities = None
        classes = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probabilities = proba.tolist()
            if hasattr(model, "classes_"):
                classes = [int(c) for c in model.classes_]

        return jsonify({
            "status": "success",
            "prediction": predicted_id,
            "college_name": college_name,
            "probabilities": probabilities,
            "classes": classes
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
