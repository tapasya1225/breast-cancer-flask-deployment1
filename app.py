import os
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model (file lives at the repo root, not in a "models/" folder)
model = joblib.load('breast_cancer_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)
    confidence = round(np.max(prediction_proba) * 100, 2)

    output = prediction[0]

    if output == 1:
        result = "Malignant Cancer Detected"
    else:
        result = "Benign Tumor"

    final_result = f"{result} | Confidence: {confidence}%"

    return render_template(
        'index.html',
        prediction_text=final_result
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)
