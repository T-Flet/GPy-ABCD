import numpy as np
from matplotlib import pyplot as plt

from GPy.models import GPRegression


# Standard Utility functions

def BIC(ll, n, k): return -2 * ll + k * np.log(n)
def AIC(ll, n, k): return 2 * (-ll + k)
def AICc(ll, n, k): return 2 * (-ll + k + (k**2 + k) / (n - k - 1)) # Assumes univariate model linear in its parameters and with normally-distributed residuals conditional upon regressors


# Kwargs passed to optimize_restarts, which passes them to optimize
#   Check comments in optimize's class AND optimization.get_optimizer for for real list of optimizers
def doGPR(X, Y, kernel, restarts, score = BIC, **kwargs):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    m = GPRegression(X, Y, kernel)

    # One fit
    # m.optimize(messages = True)

    # Best out of restarts fits
    m.optimize_restarts(num_restarts = restarts, **kwargs)

    m.plot()
    print(m.kern)
    print(f'Log-Likelihood: {m.log_likelihood()}')
    print(f'{score.__name__}: {score(*m._ordered_score_ps())}')

    plt.show()

    return m


