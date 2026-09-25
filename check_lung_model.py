import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

model_path = "models/lung_disease_densenet121_repaired.keras"

print("\nLoading repaired model...")

try:

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    print("\nMODEL LOADED SUCCESSFULLY!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

except Exception as e:

    print("\nMODEL LOAD FAILED")
    print(type(e).__name__)
    print(e)