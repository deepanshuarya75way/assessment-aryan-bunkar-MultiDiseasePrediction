import joblib

model = joblib.load("models/kidney_disease_model.pkl")

print("MODEL:")
print(model)

print("\nFEATURE NAMES:")
print(model.feature_names_in_)

print("\nPREPROCESSOR:")
preprocessor = model.named_steps["preprocessor"]

print(preprocessor)

print("\nTRANSFORMERS:")

for name, transformer, columns in preprocessor.transformers_:
    print("\nTransformer:", name)
    print("Columns:", columns)

    if hasattr(transformer, "categories_"):
        print("Categories:")
        for col, categories in zip(columns, transformer.categories_):
            print(col, "=>", categories)