import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from keras.src.saving import load_model


autoencoder = load_model("C:/Users/Marco/autoencoder_model.keras")

val_data = np.load("C:/Users/Marco/test_data.npy")

reconstructions = autoencoder.predict(val_data)

errors = np.mean(np.abs(val_data - reconstructions), axis=1)

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, alpha=0.7, color="blue", edgecolor="black", density=True)

mu, std = np.mean(errors), np.std(errors)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, "r", linewidth=2, label=f"Normal Dist (mean={mu:.4f}, stdDeviation={std:.4f})")

plt.axvline(mu + 3 * std, color="red", linestyle="dashed", linewidth=2, label="mean+ 3deviations Threshold")
plt.axvline(mu + 2 * std, color="orange", linestyle="dashed", linewidth=2, label="mean + 2deviations Threshold")

plt.title("Histogram of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()


# test for normality
shapiro_test = stats.shapiro(errors)
ks_test = stats.kstest(errors, "norm", args=(mu, std))

print(f"Shapiro-Wilk Test p-value: {shapiro_test.pvalue:.4f}")
print(f"Kolmogorov-Smirnov Test p-value: {ks_test.pvalue:.4f}")

# Decision Rule: If p value < 0.05, then the data is not normally distributed
if shapiro_test.pvalue < 0.05 or ks_test.pvalue < 0.05:
    print("Since the reconstruction errors don't follow a normal distribution we have to use te percentile based approach.")
    threshold = np.percentile(errors, 95)  # 95th percentile
    print(f"* Using 95th Percentile Threshold: {threshold:.4f}")
else:
    print("The reconstruction errors follow a normal distribution, therefore mean + 3deviations is a valid thresholding method.")
    threshold = mu + 3 * std
    print(f"* Using Mean + 3 Std Dev Threshold: {threshold:.4f}")

np.save("C:/Users/Marco/reconstruction_errors.npy", errors)
print("Reconstruction errors saved to reconstruction_errors.npy")
