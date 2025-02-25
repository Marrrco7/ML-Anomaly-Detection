import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.background import BackgroundScheduler


app = Flask(__name__)


model_path = "/home/marco/autoencoder_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


anomaly_gauge = Gauge("anomaly_score", "Anomaly score from autoencoder")


threshold = 0.1998

def update_anomaly_metric():
    try:
        data_path = "/home/marco/processed_prometheus_data_scaled.csv"
        df = pd.read_csv(data_path)

        latest_data = df.drop(columns=["timestamp"]).iloc[-1].values.astype(np.float32)
        input_data = np.expand_dims(latest_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        reconstructed = interpreter.get_tensor(output_details[0]['index'])[0]
        mse = np.mean((latest_data - reconstructed) ** 2)
        anomaly_gauge.set(mse)
        print(f"Updated anomaly metric: {mse:.5f}")
    except Exception as e:
        print(f"Error updating anomaly metric: {e}")


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(update_anomaly_metric, 'interval', seconds=15)
scheduler.start()


@app.route("/predict", methods=["GET"])
def predict():
    try:
        data_path = "/home/marco/processed_prometheus_data_scaled.csv"
        df = pd.read_csv(data_path)

        latest_data = df.drop(columns=["timestamp"]).iloc[-1].values.astype(np.float32)

        input_data = np.expand_dims(latest_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        reconstructed = interpreter.get_tensor(output_details[0]['index'])[0]

        mse = np.mean((latest_data - reconstructed) ** 2)
        anomaly_detected = mse > threshold

        anomaly_gauge.set(mse)

        return jsonify({
            "reconstruction_error": float(mse),
            "anomaly_detected": bool(anomaly_detected)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
