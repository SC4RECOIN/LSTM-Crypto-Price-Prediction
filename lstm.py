import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.utils import to_categorical
import json
import os

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


def extract_data(data):
    # obtain labels
    labels = Genlabels(data, window=25, polyorder=3).labels

    # obtain features
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=True).values

    # truncate bad values and shift label
    X = np.array([macd[30:-1], 
                  stoch_rsi[30:-1], 
                  inter_slope[30:-1],
                  dpo[30:-1], 
                  cop[30:-1]])

    X = np.transpose(X)
    labels = labels[31:]

    return X, labels

def adjust_data(X, y, split=0.8):
    # count the number of each label
    count_1 = np.count_nonzero(y)
    count_0 = y.shape[0] - count_1
    cut = min(count_0, count_1)

    # save some data for testing
    train_idx = int(cut * split)
    
    # shuffle data
    np.random.seed(42)
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    # find indexes of each label
    idx_1 = np.argwhere(y == 1).flatten()
    idx_0 = np.argwhere(y == 0).flatten()

    # grab specified cut of each label put them together 
    X_train = np.concatenate((X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
    X_test = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
    y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
    y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

    # shuffle again to mix labels
    np.random.seed(7)
    shuffle_train = np.random.permutation(X_train.shape[0])
    shuffle_test = np.random.permutation(X_test.shape[0])

    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

    return X_train, X_test, y_train, y_test

def shape_data(X, y, timesteps=10):
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if not os.path.exists('models'):
        os.mkdir('models')

    joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    return X, y

def build_model():
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

    return model

if __name__ == '__main__':
    with open('historical_data/hist_data.json') as f:
        data = json.load(f)

    # load and reshape data
    X, y = extract_data(np.array(data['close']))
    X, y = shape_data(X, y, timesteps=10)

    # ensure equal number of labels, shuffle, and split
    X_train, X_test, y_train, y_test = adjust_data(X, y)
    
    # binary encode for softmax function
    y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)

    # build and train model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=8, shuffle=True, validation_data=(X_test, y_test))
    model.save('models/lstm_model.h5')
