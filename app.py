from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load or preprocess the dataset
data = pd.read_csv("fitbit_data.csv")

data = data.sample(n=100000, random_state=42)

# Preprocessing
label_encoders = {}
categorical_columns = ['workout_type', 'weather_conditions', 'location', 'mood']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = StandardScaler()
data[['steps', 'calories_burned', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']] = scaler.fit_transform(
    data[['steps', 'calories_burned', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']]
)

# Splitting features and targets
X = data.drop(['mood', 'user_id', 'date'], axis=1)
mood_y = data['mood']
stress_y = (data['heart_rate_avg'] > 120).astype(int)  # Binary classification for stress
energy_y = data['calories_burned']  # Regression for energy level
sleep_alert_y = (data['sleep_hours'] < 6).astype(int)  # Binary classification for sleep deprivation

# Splitting data for each model
X_train_mood, X_test_mood, y_train_mood, y_test_mood = train_test_split(X, mood_y, test_size=0.2, random_state=42)
X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(X, stress_y, test_size=0.2, random_state=42)
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(X, energy_y, test_size=0.2, random_state=42)
X_train_sleep, X_test_sleep, y_train_sleep, y_test_sleep = train_test_split(X, sleep_alert_y, test_size=0.2, random_state=42)

# Train models
mood_model = RandomForestClassifier()
stress_model = RandomForestClassifier()
energy_model = RandomForestRegressor()
sleep_alert_model = RandomForestClassifier()

mood_model.fit(X_train_mood, y_train_mood)
stress_model.fit(X_train_stress, y_train_stress)
energy_model.fit(X_train_energy, y_train_energy)
sleep_alert_model.fit(X_train_sleep, y_train_sleep)

# Save models and encoders
pickle.dump(mood_model, open('mood_model.pkl', 'wb'))
pickle.dump(stress_model, open('stress_model.pkl', 'wb'))
pickle.dump(energy_model, open('energy_model.pkl', 'wb'))
pickle.dump(sleep_alert_model, open('sleep_alert_model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Flask Routes
@app.route('/predict', methods=['POST'])
def predict():
    # Load models and encoders
    mood_model = pickle.load(open('mood_model.pkl', 'rb'))
    stress_model = pickle.load(open('stress_model.pkl', 'rb'))
    energy_model = pickle.load(open('energy_model.pkl', 'rb'))
    sleep_alert_model = pickle.load(open('sleep_alert_model.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Parse input JSON
    user_data = request.json
    input_data = pd.DataFrame([user_data])

    # Encode categorical data
    for col in categorical_columns[:-1]:  # Exclude 'mood' from encoding
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale numerical data
    input_data[['steps', 'calories_burned', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']] = scaler.transform(
        input_data[['steps', 'calories_burned', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']]
    )

    # Make predictions
    mood_prediction = label_encoders['mood'].inverse_transform(mood_model.predict(input_data))[0]
    stress_prediction = "Stressed" if stress_model.predict(input_data)[0] == 1 else "Not Stressed"
    energy_prediction = round(energy_model.predict(input_data)[0], 2)

    # Rule-based check for sleep alert
    sleep_hours = user_data['sleep_hours']
    if sleep_hours < 6:
        sleep_alert = "Sleep Deprived"
    else:
        sleep_alert = "Well Rested"

    # Generate recommendations
    activity_recommendation = "Try a relaxing workout like Yoga." if mood_prediction == "Stressed" else "Keep up the good work with your routine!"
    heart_rate_anomaly = "High" if user_data['heart_rate_avg'] > 180 else "Normal"
    sedentary_alert = "Increase activity levels" if user_data['active_minutes'] < 30 else "Good activity levels"

    # Response JSON
    response = {
        "mood": mood_prediction,
        "stress_level": stress_prediction,
        "energy_level": energy_prediction,
        "sleep_alert": sleep_alert,
        "activity_recommendation": activity_recommendation,
        "heart_rate_anomaly": heart_rate_anomaly,
        "sedentary_behavior_alert": sedentary_alert
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
