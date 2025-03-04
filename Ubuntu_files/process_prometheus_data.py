import pandas as pd
import json
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

raw_csv_path = "/home/yourUser/prometheus_data.csv"
processed_csv_path = "/home/yourUser/processed_prometheus_data.csv"
scaled_csv_path = "/home/yourUser/processed_prometheus_data_scaled.csv"
scaler_path = "/home/yourUser/scaler.pkl"


if os.path.exists(raw_csv_path):
    df = pd.read_csv(raw_csv_path)
else:
    df = pd.DataFrame()

def extract_metric(json_file):
    try:
        with open(json_file) as f:
            data = json.load(f)
        if data['data']['result']:
            return float(data['data']['result'][0]['value'][1])
    except:
        return None

cpu_usage = extract_metric("/home/yourUser/cpu_usage.json")
memory_usage = extract_metric("/home/yourUser/memory_usage.json")
disk_read = extract_metric("/home/yourUser/disk_read.json")
disk_write = extract_metric("/home/yourUser/disk_write.json")
network_receive = extract_metric("/home/yourUser/net_receive.json")
network_transmit = extract_metric("/home/yourUser/net_transmit.json")

new_row = {
    'timestamp': pd.Timestamp.now(),
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'disk_read': disk_read,
    'disk_write': disk_write,
    'network_receive': network_receive,
    'network_transmit': network_transmit,
}

new_row_df = pd.DataFrame([new_row])

if df.empty:
    df = new_row_df
else:
    df = pd.concat([df, new_row_df], ignore_index=True)


df['timestamp'] = pd.to_datetime(df['timestamp'])


df['cpu_change'] = df['cpu_usage'].diff()
df['memory_change'] = df['memory_usage'].diff()
df['disk_read_speed'] = df['disk_read'].diff()
df['disk_write_speed'] = df['disk_write'].diff()
df['network_receive_speed'] = df['network_receive'].diff()
df['network_transmit_speed'] = df['network_transmit'].diff()

df['cpu_rolling'] = df['cpu_usage'].rolling(window=5).mean()
df['memory_std'] = df['memory_usage'].rolling(window=5).std()

df['cpu_mem_ratio'] = df['cpu_usage'] / df['memory_usage']
df['disk_cpu_ratio'] = df['disk_read'] / (df['cpu_usage'] + 1)

df['high_cpu'] = (df['cpu_usage'] > 80).astype(int)
df['high_memory'] = (df['memory_usage'] > 80).astype(int)


df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday


df.to_csv(raw_csv_path, index=False)


cleaned_df = df.copy()

cleaned_df.dropna(how="all", inplace=True)
cleaned_df.fillna(method="ffill", inplace=True)
cleaned_df.fillna(cleaned_df.median(), inplace=True)

for col in ['disk_read', 'disk_write', 'network_receive', 'network_transmit']:
    cleaned_df[col] = cleaned_df[col].replace(0, cleaned_df[col].median())

valid_days = pd.date_range(end=pd.Timestamp.today().date(), periods=7).strftime("%Y-%m-%d").tolist()
cleaned_df = cleaned_df[cleaned_df['timestamp'].dt.date.astype(str).isin(valid_days)]


cleaned_df.to_csv(processed_csv_path, index=False)

print(f"Data cleaned and processed successfully! Saved to: {processed_csv_path}")
print(f"Last Timestamp in CSV: {cleaned_df['timestamp'].max()}")


if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    df_scaled = cleaned_df.copy()
    

    df_scaled_no_ts = df_scaled.drop(columns=["timestamp"])

    df_scaled_no_ts[df_scaled_no_ts.columns] = scaler.transform(df_scaled_no_ts)

    df_scaled[df_scaled_no_ts.columns] = df_scaled_no_ts

    df_scaled.to_csv(scaled_csv_path, index=False)
    print(f"Scaled data updated and saved to: {scaled_csv_path}")
else:
    print("Warning: Scaler not found.")
