import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite


data_path = "/home/marco/processed_prometheus_data_scaled.csv"
df = pd.read_csv(data_path)


latest_data = df.drop(columns=['timestamp']).iloc[-1].values.astype(np.float32)

interpreter = tflite.Interpreter(model_path="/home/marco/autoencoder_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.expand_dims(latest_data, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

reconstructed = interpreter.get_tensor(output_details[0]['index'])[0]

mse = np.mean((latest_data - reconstructed) ** 2)

feature_names = df.drop(columns=['timestamp']).columns

per_feature_errors = np.abs(latest_data - reconstructed)

print("Per-feature absolute errors:")
for feature, error in zip(feature_names, per_feature_errors):
    print(f"{feature}: {error:.5f}")

threshold = 0.1998

anomaly_detected = mse > threshold

print(f"Reconstruction Error (MSE): {mse:.5f}")
print(f"🔍 Anomaly Detected: {anomaly_detected}")

if anomaly_detected:
    print("🚨 ALERT: Anomaly detected in latest Prometheus metrics!")
else:
    print("✅ System is operating normally.")
