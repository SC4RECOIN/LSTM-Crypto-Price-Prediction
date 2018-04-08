import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import sys


class gen_labels(object):
    def __init__(self, window, polyorder=3, hist='../historical_data/hist_data.npy', graph=False):
        # check for valid parameters
        try:
            if window%2 == 0: raise ValueError('Window length must be an odd positive value')
            if polyorder >= window: raise ValueError('Polyorder must be smaller than windows length')
        except ValueError as error:
            sys.exit('Error {0}'.format(error))   

        # load historic data from file
        self.hist = np.load(hist)
        self.window = window
        self.polyorder = polyorder

        self.savgol = self.apply_filter()

        if graph: self.graph()

    def apply_filter(self):
        # apply a Savitzky-Golay filter to historical prices
        return savgol_filter(self.hist, self.window, self.polyorder)

    def graph(self):
        # graph the output
        trace0 = go.Scatter(y=self.hist, name='Price')
        trace1 = go.Scatter(y=self.savgol, name='Label')

        py.plot([trace0, trace1], filename="../output/label.html")

if __name__ == '__main__':
    labels = gen_labels(window=13, polyorder=3, graph=True)
