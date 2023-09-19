import os

import keras


def get_model():
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(17,)),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1, activation='sigmoid', name='output')  # sigmoid bc binary classification
    ])
    if os.path.isfile('model.keras'):
        model.load_weights('model.keras')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
