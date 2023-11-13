import warnings
from copy import deepcopy
from GPy.models import GPRegression
from GPy.plotting import change_plotting_library

from GPy_ABCD.KernelExpressions.all import SumKE
from GPy_ABCD.KernelExpansion.kernelOperations import fit_ker_to_kex_with_params


class GPModel():
    def __init__(self, X, Y, kernel_expression = SumKE(['WN'])._initialise()):
        self.X = X
        self.Y = Y
        self.kernel_expression = kernel_expression
        self._sum_of_prods_kex = None
        self.restarts = None
        self.model = None
        self.cached_utility_function = None
        self.cached_utility_function_type = None

    # Kwargs passed to optimize_restarts, which passes them to optimize
    #   Check comments in optimize's class AND optimization.get_optimizer for real list of optimizers
    # TODO: Eventually set robust to True; see description in optimize_restarts method
    def fit(self, restarts = None, optimiser = 'lbfgsb', verbose = False, robust = False, **kwargs):
        if restarts is None:
            if self.restarts is None: raise ValueError('No restarts value specified')
        else: self.restarts = restarts
        self.model = GPRegression(self.X, self.Y, self.kernel_expression.to_kernel())
        with warnings.catch_warnings(): # Ignore known numerical warnings
            warnings.simplefilter('ignore')
            self.model.optimize_restarts(num_restarts = self.restarts, verbose = verbose, robust = robust, optimizer = optimiser, **kwargs)
        return self

    @property
    def sum_of_prods_kex(self):
        '''The canonical kernel form (the one described in .interpret).

        NOTE: this property/method can only be called after the model has been fitted.'''
        if self.model is None: raise ValueError('No parameters to insert into the kernel expression since the model has not yet been fitted')
        elif self._sum_of_prods_kex is None: self._sum_of_prods_kex = fit_ker_to_kex_with_params(self.model.kern, deepcopy(self.kernel_expression)).sum_of_prods_form()
        return self._sum_of_prods_kex

    def interpret(self):
        '''Describe the model with a few sentences (which break down the expanded kernel form, i.e. .sum_of_prods_kex).

        NOTE: this method can only be called after the model has been fitted.'''
        return self.sum_of_prods_kex.get_interpretation(sops = self._sum_of_prods_kex)

    def predict(self, X, quantiles = (2.5, 97.5), full_cov = False, Y_metadata = None, kern = None, likelihood = None, include_likelihood = True):
        mean, cov = self.model.predict(X, full_cov, Y_metadata, kern, likelihood, include_likelihood)
        qs = self.model.predict_quantiles(X, quantiles, Y_metadata, kern, likelihood)
        return {'mean': mean, 'covariance': cov, 'low_quantile': qs[0], 'high_quantile': qs[1]}

    def change_plotting_library(self, library = 'plotly_offline'):
        '''Wrapper of GPy.plotting's homonymous function;
        supported values are: 'matplotlib', 'plotly', 'plotly_online', 'plotly_offline' and 'none'.
        If 'plotly' then a 3-tuple is returned, with as 1st value the Figure object requiring a .show() to display.'''
        change_plotting_library(library)

    def plot(self): return self.model.plot()


    # Model fit objective criteria & related values:

    def _ll(self): return self.model.log_likelihood()
    def _n(self): return len(self.model.X) # number of data points
    def _k(self): return self.model._size_transformed() # number of estimated parameters, i.e. model degrees of freedom

    def _ordered_score_ps(self): return self.model, self._ll(), self._n(), self._k()

    def compute_utility(self, score_f):
        self.cached_utility_function = score_f(*self._ordered_score_ps())
        self.cached_utility_function_type = score_f.__name__
        return self.cached_utility_function


