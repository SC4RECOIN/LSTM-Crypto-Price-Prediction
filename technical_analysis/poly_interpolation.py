from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np


def poly_inter(data, degree=4, plot=False):
    # define x values for data points
    X = np.linspace(0, data.shape[0] - 1, data.shape[0])[:, np.newaxis]
    
    # define pipeline
    model = make_pipeline(PolynomialFeatures(degree), Ridge())

    # fit model
    model.fit(X, data)

    if plot: plot_poly(X, model.predict(X), data)
    
    # predict next interpolated value
    return model.predict(np.array([[data.shape[0]]]))

def plot_poly(X, y_plot, data):
    # plot interpolation
    plt.plot(X, y_plot, color='teal', linewidth=2, label="interpolation")

    # scatter plot of original points
    plt.scatter(X, data, color='navy', s=60, marker='o', label="data points")
    
    plt.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    # load the last 10 data points from file
    test_data = np.load('../historical_data/hist_data.npy')[-20:]

    # perform polynomial iterpolation
    pred = poly_inter(test_data[:-1], degree=5, plot=True)

    # compare
    print('Predicted: {0} | Actual: {1} | Degree: {2}'.format(pred, test_data[-1], 5))
