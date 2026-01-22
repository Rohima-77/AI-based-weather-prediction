from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models
scaler = pickle.load(open('scaler.pkl','rb'))
clf = pickle.load(open('weather_classifier.pkl','rb'))
reg = pickle.load(open('temperature_regressor.pkl','rb'))
kmeans = pickle.load(open('kmeans_model.pkl','rb'))
le = pickle.load(open('label_encoder.pkl','rb'))

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # Get input from form
        precipitation = float(data['precipitation'])
        temp_max = float(data['temp_max'])
        temp_min = float(data['temp_min'])
        wind = float(data['wind'])
        weather_encoded = int(data['weather_encoded'])

        input_data = [precipitation, temp_max, temp_min, wind, weather_encoded]

        # Scale for clustering
        scaled_features = scaler.transform([input_data])
        cluster = kmeans.predict(scaled_features)[0]

        # Classification
        weather_class_encoded = clf.predict([input_data])[0]
        weather_class = le.inverse_transform([weather_class_encoded])[0]

        # Regression
        reg_input = [precipitation, temp_min, wind]
        temp_pred = reg.predict([reg_input])[0]

        result = {
            'cluster': int(cluster),
            'weather_type': weather_class,
            'predicted_temperature': round(temp_pred,2)
        }

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
