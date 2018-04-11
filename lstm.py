import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import sys

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter


def extract_data():
    # obtain labels
    labels = Genlabels(window=25, polyorder=3).labels

    # obtain features
    macd = Macd(6, 12, 3).histo
    stoch_rsi = StochRsi().stoch_cross
    volume = np.load('historical_data/hist_volume.npy')
    interpolation = PolyInter(progress_bar=True).values

    # truncate bad values and shift label
    X = np.array([macd[30:-1], stoch_rsi[30:-1], volume[30:-1], interpolation[30:-1]])
    labels = labels[31:]

    try:
        # make sure data is the same length
        if macd.shape[0] != stoch_rsi.shape[0] \
        or macd.shape[0] != interpolation.shape[0] \
        or macd.shape[0] != volume.shape[0]: 
            raise Exception('Data is not the same length')
    except Exception as error:
            sys.exit('Error: {0}'.format(error))

    return np.transpose(X), labels

def split(X, y, split=0.8):
    # split data
    idx = int(X.shape[0] * split)
    X_train, X_test = X[:idx], X[idx:] 
    y_train, y_test = y[:idx], y[idx:]

    return X_train, X_test, y_train, y_test

def shape_data(X, y, timesteps=20, split=0.8):
    # scale data without looking into test data
    to_fit_idx = int(X.shape[0] * split) 
    scaler = StandardScaler().fit(X[:to_fit_idx])
    X = scaler.transform(X)

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    # shuffle data
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    return X, to_categorical(y, 2)

def build_model(X, y, val_x, val_y):
    # first layer
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # fourth layer and output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=30, batch_size=8, shuffle=True, validation_data=(val_x, val_y))

    return model

if __name__ == '__main__':
    # load data and shape it
    X, y = extract_data()
    X, y = shape_data(X, y)
    X_train, X_test, y_train, y_test = split(X, y)

    model = build_model(X_train, y_train, X_test, y_test)
