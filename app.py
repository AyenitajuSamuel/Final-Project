from flask import Flask, render_template, request
import pandas as pd
import joblib

# ----------------------------
# Load trained ensemble + features
# ----------------------------
saved_obj = joblib.load("soft_voting_ensemble.pkl")  # or stacking_ensemble.pkl
model = saved_obj["model"]
feature_names = saved_obj["features"]

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values dynamically
        input_data = [float(request.form.get(f)) for f in feature_names]

        # Convert to DataFrame with feature names
        features = pd.DataFrame([input_data], columns=feature_names)

        # Run prediction
        prob = model.predict_proba(features)[0][1]
        prediction = "High Risk of CAD" if prob > 0.5 else "Low Risk of CAD"

       # inside predict()
        return render_template(
            "result.html",
            prediction=prediction,
            probability=float(round(prob, 3))  # ensure numeric for meter width
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
