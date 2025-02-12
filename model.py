from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to avoid frontend request issues

# Load the trained ML model
with open("RFmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved encoders for categorical variables
with open("soil_encoder.pkl", "rb") as f:
    soil_encoder = pickle.load(f)

with open("crop_encoder.pkl", "rb") as f:
    crop_encoder = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have index.html in templates folder
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        print("Received Data:", data)  # Debugging line

        # Extract values
        soil_type = data.get("soil", None)  # Safer way to extract
        crop_type = data.get("crop", None)

        if soil_type is None or crop_type is None:
            return jsonify({"error": "Missing soil_type or crop_type in request"}), 400

        # Continue with encoding and prediction...


        
        # Extract values
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        moisture = float(data["moisture"])
        soil_type = data["soil_type"]
        crop_type = data["crop_type"]
        nitrogen = float(data["nitrogen"])
        phosphorous = float(data["phosphorous"])
        potassium = float(data["potassium"])

        # Encode categorical variables
        if soil_type in soil_encoder.classes_:
            soil_type_encoded = soil_encoder.transform([soil_type])[0]
        else:
            return jsonify({"error": f"Unknown soil type: {soil_type}"}), 400

        if crop_type in crop_encoder.classes_:
            crop_type_encoded = crop_encoder.transform([crop_type])[0]
        else:
            return jsonify({"error": f"Unknown crop type: {crop_type}"}), 400

        # Prepare input features
        features = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, phosphorous, potassium]])

        # Predict the fertilizer recommendation
        prediction = model.predict(features)
        
        # Return the result
        return jsonify({"fertilizer": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Added host and port for better accessibility
