import keras_tuner as kt
from tensorflow import keras
import numpy as np


train_data = np.load("C:/Users/yourUser/train_data.npy")
test_data = np.load("C:/Users/yourUser/test_data.npy")

def build_model(hp):
    input_dim = train_data.shape[1]
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        units=hp.Int('units1', min_value=8, max_value=32, step=4),
        activation='relu',
        input_shape=(input_dim,)
    ))

    model.add(keras.layers.Dense(
        units=hp.Int('units2', min_value=4, max_value=16, step=2),
        activation='relu'
    ))

    model.add(keras.layers.Dense(4, activation='relu'))

    model.add(keras.layers.Dense(
        units=hp.Int('units3', min_value=4, max_value=16, step=2),
        activation='relu'
    ))

    model.add(keras.layers.Dense(
        units=hp.Int('units4', min_value=8, max_value=32, step=4),
        activation='relu'
    ))

    model.add(keras.layers.Dense(input_dim, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse')
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='autoencoder_tuning'
)

tuner.search(train_data, train_data, epochs=50, validation_data=(test_data, test_data))

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hp.values)

best_model = tuner.get_best_models(num_models=1)[0]
print("Best Model Summary:")
best_model.summary()
