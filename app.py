import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model File Path
MODEL_PATH = "Clgname_model_pkl"

# Load the Scikit-Learn Model
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found in the root directory!")

# HTML & JavaScript Template Embedded Directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Prediction Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js for Visualization -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-100 font-sans leading-normal tracking-normal">

    <!-- Header / Navbar -->
    <nav class="bg-indigo-600 p-4 text-white shadow-lg">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">🎓 College Predictor Dashboard</h1>
            <span class="text-sm bg-indigo-800 px-3 py-1 rounded-full">Flask ML App</span>
        </div>
    </nav>

    <div class="container mx-auto my-6 px-4">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

            <!-- Form Panel -->
            <div class="bg-white p-6 rounded-lg shadow-md lg:col-span-1">
                <h2 class="text-xl font-bold text-gray-800 mb-4 border-b pb-2">Candidate Details</h2>
                <form id="predictorForm" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Merit Number</label>
                        <input type="number" id="merit_number" required class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500" placeholder="e.g. 12000">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">MHTCET Percentile</label>
                        <input type="number" step="0.01" id="mhtcet_percentile" required class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500" placeholder="e.g. 95.5">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Candidate Name Code</label>
                        <input type="number" id="name" value="1" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                    </div>
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Gender Code</label>
                            <input type="number" id="gender" value="0" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Category Code</label>
                            <input type="number" id="category" value="1" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Seat Alloted Code</label>
                            <input type="number" id="seat_alloted" value="1" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Course Code</label>
                            <input type="number" id="course_name" value="1" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Seat Number Code</label>
                        <input type="number" id="seat_number" value="101" class="w-full mt-1 p-2 border rounded-md focus:ring-2 focus:ring-indigo-500">
                    </div>

                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 rounded-md transition duration-200">
                        Predict Allocation
                    </button>
                </form>
            </div>

            <!-- Output Panel -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Result Box -->
                <div class="bg-white p-6 rounded-lg shadow-md border-l-4 border-indigo-600 flex justify-between items-center">
                    <div>
                        <h3 class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Predicted College ID / Result</h3>
                        <p id="predictedOutput" class="text-3xl font-extrabold text-gray-800 mt-1">--</p>
                    </div>
                    <div class="text-4xl text-indigo-500">🏫</div>
                </div>

                <!-- Chart Analytics -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h3 class="text-lg font-bold text-gray-800 mb-4">Probability Analytics</h3>
                    <div class="relative h-64">
                        <canvas id="probabilityChart"></canvas>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Script to handle AJAX Predict Request -->
    <script>
        let chartInstance = null;

        document.getElementById('predictorForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const payload = {
                merit_number: document.getElementById('merit_number').value,
                mhtcet_percentile: document.getElementById('mhtcet_percentile').value,
                name: document.getElementById('name').value,
                gender: document.getElementById('gender').value,
                category: document.getElementById('category').value,
                seat_alloted: document.getElementById('seat_alloted').value,
                course_name: document.getElementById('course_name').value,
                seat_number: document.getElementById('seat_number').value
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if(result.status === "success") {
                document.getElementById('predictedOutput').innerText = "Class/College: " + result.prediction;
                if (result.probabilities) {
                    renderChart(result.probabilities);
                }
            } else {
                alert("Error: " + result.message);
            }
        });

        function renderChart(probabilities) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            const labels = probabilities.map((_, index) => `Class ${index}`);

            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Prediction Confidence / Probability',
                        data: probabilities,
                        backgroundColor: 'rgba(79, 70, 229, 0.6)',
                        borderColor: 'rgba(79, 70, 229, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 1 }
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
    """Renders the HTML Dashboard UI directly from the inline template string."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to receive form values and return predictions."""
    if not model:
        return jsonify({"status": "error", "message": "Model pickle file not loaded properly."}), 500

    try:
        data = request.get_json()

        # Feature order expected by the model
        merit_number = float(data.get("merit_number", 0))
        mhtcet_percentile = float(data.get("mhtcet_percentile", 0))
        name = float(data.get("name", 0))
        gender = float(data.get("gender", 0))
        category = float(data.get("category", 0))
        seat_alloted = float(data.get("seat_alloted", 0))
        course_name = float(data.get("course_name", 0))
        seat_number = float(data.get("seat_number", 0))

        # Format inputs as a NumPy array
        input_features = np.array([[
            merit_number,
            mhtcet_percentile,
            name,
            gender,
            category,
            seat_alloted,
            course_name,
            seat_number
        ]])

        # Model Prediction
        prediction = model.predict(input_features)
        
        # Calculate class probabilities (if supported by model)
        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_features)[0]
            probabilities = proba.tolist()

        return jsonify({
            "status": "success",
            "prediction": int(prediction[0]),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
