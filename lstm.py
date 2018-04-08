import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import plotly.offline as py
import plotly.graph_objs as go

from technical_analysis.generate_labels import Genlabels


def extract_data():
    labels = Genlabels(window=25, polyorder=3, graph=True)

    return labels.savgol_deriv

def build_model(inputs, step, nodes=64):

    model = Sequential()
    model.add(LSTM(
        step,
        input_shape=(inputs, step),
        return_sequences=True))
    model.add(LSTM(nodes, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="rmsprop")

    return model

if __name__ == '__main__':
