import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from plotnine import ggplot, geom_line, aes

from GPy.models import GPRegression


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2: return {t1: [k1, k2]}
    else: return {t1: k1, t2: k2}



## GPy Model Scoring

def score_ps(m):
    ll = m.log_likelihood()
    n = len(m.X) # number of data points
    k = m._size_transformed() # number of estimated parameters, i.e. model degrees of freedom
    return ll, n, k

def BIC(ll, n, k): return -2 * ll + k * np.log(n)
def AIC(ll, n, k): return 2 * (-ll + k)
def AICc(ll, n, k): return 2 * (-ll + k + (k**2 + k) / (n - k - 1))


# Kwargs passed to optimize_restarts, which passes them to optimize
#   Check comments in optimize's class AND optimization.get_optimizer for for real list of optimizers
def doGPR(X, Y, kernel, restarts, score = BIC, **kwargs):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    m = GPRegression(X, Y, kernel)

    # One fit
    # m.optimize(messages=True)

    # Best out of restarts fits
    m.optimize_restarts(num_restarts = restarts, **kwargs)

    m.plot()
    print(m.kern)
    print(f'Log-Likelihood: {m.log_likelihood()}')
    print(f'{score.__name__}: {score(*score_ps(m))}')

    plt.show()

    return m


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
