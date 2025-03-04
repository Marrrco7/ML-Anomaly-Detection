import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/yourUser/autoencoder_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = "C:/Users/yourUser/autoencoder_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted to TensorFlow Lite")
print(f"TFLite model saved at: {tflite_model_path}")
