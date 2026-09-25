import joblib
import pandas as pd

# Load Liver Disease model
model = joblib.load("models/liver_disease_model.pkl")

# Test patient
patient = pd.DataFrame([{
    "age": 40,
    "gender": 1,
    "total_bilirubin": 1.0,
    "direct_bilirubin": 0.3,
    "alkphos": 200,
    "sgpt": 30,
    "sgot": 30,
    "total_proteins": 6.5,
    "albumin": 3.5,
    "ag_ratio": 1.0
}])

# Prediction
prediction = model.predict(patient)[0]

# Probabilities
probabilities = model.predict_proba(patient)[0]

no_disease_probability = probabilities[0]
disease_probability = probabilities[1]

# Prediction text
if prediction == 1:
    prediction_text = "Liver Disease"
else:
    prediction_text = "No Liver Disease"

print("\n================================")
print("LIVER DISEASE RESULT")
print("================================")

print("\nPrediction:")
print(prediction_text)

print("\nNo Disease Probability:")
print(f"{no_disease_probability * 100:.2f}%")

print("\nDisease Probability:")
print(f"{disease_probability * 100:.2f}%")