import numpy as np
from matplotlib import pyplot as plt

from GPy.models import GPRegression

from GPy_ABCD.Models.model import GPModel


# Standard Utility functions

def BIC(ll, n, k): return -2 * ll + k * np.log(n)
def AIC(ll, n, k): return 2 * (-ll + k)
def AICc(ll, n, k): return 2 * (-ll + k + (k**2 + k) / (n - k - 1)) # Assumes univariate model linear in its parameters and with normally-distributed residuals conditional upon regressors


# Allowed GPy optimisers ('lbfgsb' is the default)
#   NOTE:
#       - Names autocomplete from initial input, e.g. 'lb' will select 'lbfgsb' with no warning
#       - 'adadelta', 'rprop' and 'adam' require the climin library
GPy_optimisers = ['lbfgsb', 'org-bfgs', 'fmin_tnc', 'scg', 'simplex', 'adadelta', 'rprop', 'adam']


def model_printout(m):
    print(m.kernel_expression)
    print(m.model.kern)
    print(f'Log-Lik: {m.model.log_likelihood()}')
    print(f'{m.cached_utility_function_type}: {m.cached_utility_function}')
    m.model.plot()
    print(m.interpret())


# Kwargs passed to optimize_restarts, which passes them to optimize
#   Check comments in optimize's class AND optimization.get_optimizer for for real list of optimizers
def fit_kex(X, Y, kex, restarts, score = BIC, **kwargs):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    m = GPModel(X, Y, kex._initialise())
    m.fit(restarts, verbose = True, **kwargs)
    m.compute_utility(score)

    return m


# Kwargs passed to optimize_restarts, which passes them to optimize
#   Check comments in optimize's class AND optimization.get_optimizer for for real list of optimizers
def fit_GPy_kern(X, Y, kernel, restarts, score = BIC, **kwargs):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    m = GPRegression(X, Y, kernel)
    m.optimize_restarts(num_restarts = restarts, **kwargs)

    m.plot()
    print(m.kern)
    print(f'Log-Likelihood: {m.log_likelihood()}')
    print(f'{score.__name__}: {score(m.log_likelihood(), len(X), m._size_transformed())}')

    plt.show()

    return m


