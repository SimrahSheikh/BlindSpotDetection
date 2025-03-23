import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from blind_spot_model import BlindSpotDetectionModel
from risk_assessment import get_warning_level

# Load dataset
data = pd.read_csv("blind_spot_detection_data.csv")

# Split features and labels
X = data.drop(columns=["Risk Level"])  # Features
y = data["Risk Level"]  # Labels

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = BlindSpotDetectionModel()
model.train_models(X_train, y_train)
model.evaluate_models(X_train, y_train, X_test, y_test)

# Save trained model
model.save_models()

# Load trained model
model.load_models()

# Simulated live sensor data
test_data = [
    [27.5, -41.2, -4.0, 20.7, -18.1, 14.9, 6.6, 15.2, 6.6, 17.9, -5.7],  # Example input
    [30.0, -35.0, -5.5, 22.3, -16.8, 13.5, 8.0, 14.8, 7.2, 18.3, -4.8],
    [35.5, -50.1, -6.8, 24.6, -20.2, 10.2, 4.5, 12.9, 5.8, 16.5, -6.3]
]

# Run predictions
for sensor_input in test_data:
    risk_level = model.predict(sensor_input)
    print(get_warning_level(risk_level))
