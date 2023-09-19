import keras
import pandas as pd

from model import get_model
from preprocess import preprocess


data = preprocess(pd.read_csv('train.csv'))
ans = data['Survived']
data = data.drop(['Survived'], axis=1)
print(data.head())
model = get_model()
print(model.summary())
checkpoint = keras.callbacks.ModelCheckpoint('model.keras', save_best_only=True, save_weights_only=True, mode='max', monitor='val_accuracy')
model.fit(data, ans, epochs=1000, callbacks=[checkpoint], shuffle=True, batch_size=32, validation_split=0.2) # shuffling data is important
# fact: early stop is bad bc it'll improve in the long term
