import joblib
import pandas as pd

from guidance.guidance_engine import GuidanceEngine


# ============================================
# 1. Load Heart Disease model
# ============================================

model = joblib.load(
    "models/heart_disease_model.pkl"
)


# ============================================
# 2. Create patient input
# ============================================

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


# ============================================
# 3. Make prediction
# ============================================

prediction = model.predict(patient)[0]

probabilities = model.predict_proba(patient)[0]


# ============================================
# 4. Get disease probability
# ============================================

disease_probability = probabilities[1]

no_disease_probability = probabilities[0]


# ============================================
# 5. Convert prediction to text
# ============================================

if prediction == 1:
    prediction_text = "Heart Disease"
else:
    prediction_text = "No Heart Disease"


# ============================================
# 6. Load Guidance Engine
# ============================================

engine = GuidanceEngine()


# ============================================
# 7. Get guidance
# ============================================

guidance = engine.get_guidance(
    disease_id=1,
    probability=disease_probability,
    inputs=patient
)


# ============================================
# 8. Display result
# ============================================

print("\n================================")
print("HEART DISEASE RESULT")
print("================================")

print("\nPrediction:")
print(prediction_text)

print("\nNo Disease Probability:")
print(f"{no_disease_probability:.2%}")

print("\nDisease Probability:")
print(f"{disease_probability:.2%}")


print("\n================================")
print("PROBABILITY LEVEL")
print("================================")

print(guidance["probability_level"])

print("\nAction:")
print(guidance["probability_action"])


print("\n================================")
print("WHAT TO CONSUME")
print("================================")

for item in guidance["consume"]:
    print("-", item)


print("\n================================")
print("WHAT TO LIMIT")
print("================================")

for item in guidance["limit"]:
    print("-", item)


print("\n================================")
print("LIFESTYLE")
print("================================")

for item in guidance["lifestyle"]:
    print("-", item)


print("\n================================")
print("INPUT-SPECIFIC OBSERVATIONS")
print("================================")

for item in guidance["input_flags"]:
    print("-", item["message"])


print("\n================================")
print("WHEN TO SEEK CARE")
print("================================")

for item in guidance["seek_care"]:
    print("-", item)


print("\n================================")
print("SOURCES")
print("================================")

for source in guidance["sources"]:
    print(
        "-",
        source["organization"],
        "-",
        source["title"]
    )