import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import sys


class Genlabels(object):
    def __init__(self, window, polyorder=3, hist='historical_data/hist_data.npy', graph=False):
        # check for valid parameters
        try:
            if window%2 == 0: raise ValueError('Window length must be an odd positive value')
            if polyorder >= window: raise ValueError('Polyorder must be smaller than windows length')
        except ValueError as error:
            sys.exit('Error: {0}'.format(error))   

        # load historic data from file
        self.hist = np.load(hist)
        self.window = window
        self.polyorder = polyorder

        # filter data and generate labels
        self.savgol = self.apply_filter(deriv=0)
        self.savgol_deriv = self.apply_filter(deriv=1)

        if graph: self.graph()

    def apply_filter(self, deriv):
        # apply a Savitzky-Golay filter to historical prices
        return savgol_filter(self.hist, self.window, self.polyorder, deriv=deriv)          

    def graph(self):
        # graph the labels
        trace0 = go.Scatter(y=self.hist, name='Price')
        trace1 = go.Scatter(y=self.savgol, name='Filter')
        trace2 = go.Scatter(y=self.savgol_deriv, name='Derivative', yaxis='y2')
        data = [trace0, trace1, trace2]

        layout = go.Layout(
            title='Labels',
            yaxis=dict(
                title='USDT value'
            ),
            yaxis2=dict(
                title='Derivative of Filter',
                overlaying='y',
                side='right'
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='../docs/label.html')

if __name__ == '__main__':
    labels = Genlabels(window=25, polyorder=3, hist='../historical_data/hist_data.npy', graph=True)
