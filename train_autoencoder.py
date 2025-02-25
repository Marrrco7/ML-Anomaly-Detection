import numpy as np
from tensorflow import keras


train_data = np.load("C:/Users/Marco/train_data.npy")
test_data = np.load("C:/Users/Marco/test_data.npy")

input_dim = train_data.shape[1]
model = keras.Sequential([
    keras.layers.Dense(14, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

history = model.fit(
    train_data, train_data,
    epochs=100,
    batch_size=32,
    validation_data=(test_data, test_data)
)

model.save("C:/Users/Marco/autoencoder_model.keras")
print("Autoencoder Model Trained and Saved")
