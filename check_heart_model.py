import joblib

model = joblib.load("models/heart_disease_model.pkl")

print("Model type:")
print(type(model))

print("\nModel:")
print(model)

print("\nHas predict_proba:")
print(hasattr(model, "predict_proba"))