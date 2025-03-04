import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

file_path = "C:/Users/yourUser/processed_prometheus_data_copy.csv"
df = pd.read_csv(file_path).drop(columns=["timestamp"])

scaler = MinMaxScaler()
scaler.fit(df)

joblib.dump(scaler, "C:/Users/yourUser/scaler.pkl")
print(f"New scaler trained on {scaler.n_features_in_} features and saved!")
