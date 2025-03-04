import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reconstruction_errors = np.load("C:/Users/yourUser/reconstruction_errors.npy")

percentiles = [95, 99, 99.5]
thresholds = {p: np.percentile(reconstruction_errors, p) for p in percentiles}

anomaly_counts = {p: np.sum(reconstruction_errors > t) for p, t in thresholds.items()}

plt.figure(figsize=(10, 5))
sns.histplot(reconstruction_errors, bins=30, kde=True, color="blue")

for p, t in thresholds.items():
    plt.axvline(t, color="red" if p >= 99 else "orange", linestyle="--", label=f"{p}th Percentile ({t:.4f})")

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Histogram of Reconstruction Errors with Percentile-Based Thresholds")
plt.legend()
plt.show()

for p in percentiles:
    print(f"{p}th Percentile Threshold: {thresholds[p]:.4f}")
    print(f"Number of Anomalies Detected: {anomaly_counts[p]}")
