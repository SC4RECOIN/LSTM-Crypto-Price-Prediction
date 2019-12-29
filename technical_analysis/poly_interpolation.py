from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np


class PolyInter(object):
    def __init__(self, data, degree=4, pd=20, plot=False, progress_bar=False):
        self.degree = degree
        self.pd = pd
        self.progress = progress_bar
        
        # required matplotlib
        self.plot = plot

        self.values = self.calc_poly(data)

    def calc_poly(self, data):
        pred_inter = [0] * (self.pd - 1)

        # iterate through prices to find interpolations
        for i in range(self.pd, len(data) + 1):
            pred_inter.append(self.poly_inter(data[i - self.pd:i]))
            
            # monitor progress
            if self.progress: print('\rInterpolation progress: {0:.2f}%'.format(i / len(data) * 100), end='')
            if i == len(data): print('')

        return np.array(pred_inter)

    def poly_inter(self, data):
        # define x values for data points
        X = np.linspace(0, data.shape[0] - 1, data.shape[0])[:, np.newaxis]
        
        # define pipeline and fit model
        model = make_pipeline(PolynomialFeatures(self.degree), Ridge())
        model.fit(X, data)

        if self.plot: plot_poly(X, model.predict(X), data)
        
        # predict next interpolated value
        last = model.predict(np.array([[data.shape[0] - 1]]))
        pred = model.predict(np.array([[data.shape[0]]]))

        # return slope of last point
        return pred[0]/last[0]

def plot_poly(X, y_plot, data):
    # plot interpolation
    plt.plot(X, y_plot, color='teal', linewidth=2, label="interpolation")

    # scatter plot of original points
    plt.scatter(X, data, color='navy', s=60, marker='o', label="data points")
    
    plt.legend(loc='lower left')
    plt.show()
