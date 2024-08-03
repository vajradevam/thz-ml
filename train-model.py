import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data_aman/data.csv")
df = df.sample(frac=1).reset_index(drop=True)

xcols = ['a', 'inl', 'pl', 'pw']
ycols = ['freq', 's11', 'gain']

data_x = df[xcols].values
data_y = df[ycols].values

def train_test_split(data: np.ndarray, tsp=0.8):
    train_size = round(tsp * len(data))
    return data[:train_size], data[train_size:]

train_x, test_x = train_test_split(data_x)
train_y, test_y = train_test_split(data_y)

from Scaler import MinMaxScaler

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(train_x)
scaler_y.fit(train_y)

train_x_t = scaler_x.transform(train_x)
test_x_t = scaler_x.transform(test_x)

train_y_t = scaler_y.transform(train_y)
test_y_t = scaler_y.transform(test_y)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

import pickle

class HistorySaver(Callback):
    def __init__(self, history_file):
        self.history_file = history_file
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.history.append(logs)

    def on_train_end(self, logs=None):
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.history, f)

model = Sequential()

model.add(Dense(64, input_dim=train_x_t.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history_saver = HistorySaver('./tmp/training_history.pkl')
checkpoint = ModelCheckpoint('./tmp/model_checkpoint.h5', save_best_only=False)

callbacks = [history_saver, checkpoint]

history = model.fit(
    train_x_t, train_y_t,
    validation_split=0.15,
    epochs=10000,
    callbacks=callbacks,
    verbose=0
)
