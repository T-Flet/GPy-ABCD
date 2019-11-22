import numpy as np
import pylab as pb
from matplotlib import pyplot as plt
import pandas as pd
from plotnine import ggplot, geom_line, aes
from GPy.models import GPRegression


def sampleCurves(k):
    """
    Plot sample curves from a given kernel
    """
    X = np.linspace(-3., 3., 500)  # 500 points evenly spaced over [3,3]
    X = X[:, None]   # reshape X to make it n*D
    mu = np.zeros(500)  # vector of the means
    C = k.K(X, X)    # covariance matrix
    # Generate 20 sample path with mean mu and covariance C
    Z = np.random.multivariate_normal(mu, C, 20).T

    df = pd.DataFrame(np.concatenate((X, Z), 1), columns=['X'] + list(range(Z.shape[1])))
    print(ggplot(pd.melt(df, id_vars=['X'], value_vars=list(range(Z.shape[1]))[1:])) +
            geom_line(aes('X', 'value', color='variable')))


def sampleCurvesNonGG(k):
    """
    Plot sample curves from a given kernel
    """
    X = np.linspace(-3., 3., 500)  # 500 points evenly spaced over [-3,3]
    X = X[:, None]   # reshape X to make it n*D
    mu = np.zeros(500)  # vector of the means
    C = k.K(X, X)    # covariance matrix
    # Generate 20 sample path with mean mu and covariance C
    Z = np.random.multivariate_normal(mu, C, 20)

    pb.figure() # open new plotting window
    for i in range(20): pb.plot(X[:], Z[i, :])
    plt.show()


def doGPR(X, Y, kernel, restarts):
    m = GPRegression(X, Y, kernel)

    # One fit
    # m.optimize(messages=True)

    # Best out of restarts fits
    m.optimize_restarts(num_restarts = restarts)

    m.plot()
    print(m.kern)
    print(m.log_likelihood())

    plt.show()

    return m


# TODO:
#   Add two functions to quickly generate changepoint and changewindow data from two given formulae
