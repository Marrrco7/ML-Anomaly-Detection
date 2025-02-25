import pandas as pd
import numpy as np

file_path = "C:/Users/Marco/processed_prometheus_data_scaled.csv"
df = pd.read_csv(file_path, parse_dates=['timestamp'])

#
features = df.drop(columns=['timestamp']).columns

train_df = df[(df['timestamp'] >= '2025-02-21') & (df['timestamp'] < '2025-02-24')]
test_df = df[df['timestamp'] < '2025-02-21']


X_train = train_df[features].values
X_test = test_df[features].values

train_file = "C:/Users/Marco/train_data.npy"
test_file = "C:/Users/Marco/test_data.npy"
np.save(train_file, X_train)
np.save(test_file, X_test)

print(f"Data split completed! Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")
print(f"Training data saved at: {train_file}")
print(f"Validation data saved at: {test_file}")
