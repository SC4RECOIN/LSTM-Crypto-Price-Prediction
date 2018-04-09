import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd


def extract_data():
    # obtain labels
    labels = Genlabels(window=25, polyorder=3, graph=True)

    # obtain features
    macd = Macd(6, 12, 3).histo
    volume = np.load('historical_data/hist_volume.npy')

    return labels.savgol_deriv

def build_model(inputs, step):

    model = Sequential()

    # first layer
    model.add(LSTM(32, input_shape=(inputs, step), return_sequences=True))
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))

    # third layer and output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="rmsprop")

    return model

if __name__ == '__main__':
    X, y = extract_data()
