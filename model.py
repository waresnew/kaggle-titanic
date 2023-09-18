import os

import keras


def get_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(11,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', name='output')  # sigmoid bc binary classification
    ])
    if os.path.isfile('model.keras'):
        model.load_weights('model.keras')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
