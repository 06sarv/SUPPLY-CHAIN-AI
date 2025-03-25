import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model and correct label encoders
model = joblib.load("shipping_model.pkl")
weather_enc = joblib.load("weather_label_encoder.pkl")  # Weather encoder
status_enc = joblib.load("status_label_encoder.pkl")  # Status encoder (new)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Shipping Delay Predictor API is Running! Use /predict endpoint."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_weather = data["Weather_Condition"].strip().lower()  # Normalize input

        # Create a lowercase mapping of available weather conditions
        weather_mapping = {w.lower(): w for w in weather_enc.classes_}

        # Validate Weather Condition
        if input_weather not in weather_mapping:
            return jsonify({"error": f"Unrecognized weather condition: {data['Weather_Condition']}. Expected one of {list(weather_mapping.keys())}"})

        # Transform weather condition using the correct encoder
        encoded_weather = weather_enc.transform([weather_mapping[input_weather]])[0]

        # Prepare input data
        input_data = pd.DataFrame([[
            data["Supplier_Performance"],
            data["Past_Delays"],
            encoded_weather,
            data["Distance_km"]
        ]], columns=["Supplier_Performance", "Past_Delays", "Weather_Condition", "Distance_km"])

        # Make prediction
        encoded_prediction = model.predict(input_data)[0]

        # Decode prediction to actual label ("On-Time" or "Delayed")
        status = status_enc.inverse_transform([encoded_prediction])[0]

        return jsonify({"prediction": status})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
