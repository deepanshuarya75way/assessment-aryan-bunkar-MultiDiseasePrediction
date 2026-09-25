import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/heart_disease_model.pkl")

# Patient data
patient = pd.DataFrame([{
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6
}])

# Prediction
prediction = model.predict(patient)[0]

# Probabilities
probabilities = model.predict_proba(patient)[0]

print("Prediction:", prediction)

print("\nProbabilities:")
print(probabilities)