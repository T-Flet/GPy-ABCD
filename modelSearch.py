from operator import methodcaller, attrgetter
import warnings
from GPy.models import GPRegression
from KernelExpansion.grammar import *


class GPModel():
    def __init__(self, X, Y, kernel_expression = SumKE(['WN'])._initialise()):
        self.X = X
        self.Y = Y
        self.kernel_expression = kernel_expression
        self.restarts = None
        self.model = None
        self.cached_utility_function = None
        self.cached_utility_function_type = None

    def fit(self, restarts = None):
        if restarts is None:
            if self.restarts is None: raise ValueError('No restarts value specified')
        else: self.restarts = restarts
        self.model = GPRegression(self.X, self.Y, self.kernel_expression.to_kernel())
        self.model.optimize_restarts(num_restarts = self.restarts, verbose = False)
        return self


    # Model fit objective criteria & related values:

    def _ll(self): return self.model.log_likelihood()
    def _n(self): return len(self.model.X) # number of data points
    def _k(self): return self.model._size_transformed() # number of estimated parameters, i.e. model degrees of freedom

    def AIC(self):
        self.cached_utility_function = 2 * (-self._ll() + self._k())
        self.cached_utility_function_type = 'AIC'
        return self.cached_utility_function

    def AICc(self): # Assumes univariate model linear in its parameters and with normally-distributed residuals conditional upon regressors
        self.cached_utility_function = 2 * (-self._ll() + self._k() + (self._k()**2 + self._k()) / (self._n() - self._k() - 1))
        self.cached_utility_function_type = 'AICc'
        return self.cached_utility_function

    def BIC(self):
        self.cached_utility_function = -2 * self._ll() + self._k() * np.log(self._n())
        self.cached_utility_function_type = 'BIC'
        return self.cached_utility_function


# Model Testing functions

def find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules, restarts = 5,
                    utility_function = 'BIC', depth = 2, buffer = 5, verbose = False):
    if verbose: print(f'Testing {depth} layers of model expansion starting from: {start_kernel}\nModels are fitted with {restarts} random restarts and scored by {utility_function}\nOnly the {buffer} best models proceed to the next layer of expansion')
    tested_k_exprs = [start_kernel]
    tested_models = [[GPModel(X, Y, start_kernel).fit(restarts)]]
    methodcaller(utility_function)(tested_models[0][0])
    with warnings.catch_warnings(): # Ignore known numerical warnings
        warnings.simplefilter("ignore")
        for d in range(1, depth + 1):
            tested_models.append([]) # tested_models[d] = []
            model_buffer = tested_models[d - 1][:buffer] if len(tested_models[d - 1]) >= buffer else tested_models[d - 1]
            for model in model_buffer: # This should be parallelised
                for kex in expand(model.kernel_expression, p_rules.values()):
                    if kex not in tested_k_exprs:
                        mod = GPModel(X, Y, kex).fit(restarts)
                        tested_k_exprs.append(kex)
                        tested_models[d].append(mod)
            tested_models[d].sort(key = methodcaller(utility_function))
            if verbose: print(f'Best depth-{d} model: {tested_models[d][0].kernel_expression}')
        sorted_models = sorted(flatten(tested_models), key = attrgetter('cached_utility_function'))
        if verbose: print(f'Best model overall: {sorted_models[0].kernel_expression}')
    return sorted_models[:5], tested_models, tested_k_exprs






## TESTING

import numpy as np

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


X = np.linspace(-10, 10, 101)[:, None]

Y = np.cos( (X - 5) / 2 )**2 * 7 + np.random.randn(101, 1) * 1 #- 100

# from Util.util import doGPR
# doGPR(X, Y, PER + C, 10)


best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules,
                                                 restarts = 2, utility_function = 'BIC', depth = 2, buffer = 5, verbose= True)

from matplotlib import pyplot as plt
for bm in best_mods:
    print(bm.kernel_expression)
    print(bm.model.kern)
    print(bm.model.log_likelihood())
    print(bm.cached_utility_function)
    bm.model.plot()

plt.show()
