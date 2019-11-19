from operator import methodcaller, attrgetter
import warnings
from GPy.models import GPRegression
import numpy as np
from KernelExpansion.grammar import *
from multiprocessing import Pool, cpu_count


## Model class

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
        with warnings.catch_warnings(): # Ignore known numerical warnings
            warnings.simplefilter("ignore")
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



## Model Sarch

# NOTE: Any code that calls (however many nested levels in) fit_model_list, needs to be within a
#   if __name__ == '__main__':
#   preamble in order to work on Windows. Note that this includes find_best_model, and hence any call to this project.

# TODO:
#  Decide whether to have a set of initial common models before or instead of WN expansion in order to return something quickly
#   (this is related to the different sets of production rules possibly being applied at different depth levels)

def fit_one_model(X, Y, kex, restarts): return GPModel(X, Y, kex).fit(restarts)
def fit_model_list(X, Y, k_exprs, restarts = 5):
    with Pool() as pool: return pool.starmap_async(fit_one_model, [(X, Y, kex, restarts) for kex in k_exprs], int(len(k_exprs) / cpu_count()) + 1).get()


def find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules_all, restarts = 5,
                    utility_function = 'BIC', depth = 2, buffer = 5, verbose = False):
    if verbose: print(f'Testing {depth} layers of model expansion starting from: {start_kernel}\nModels are fitted with {restarts} random restarts and scored by {utility_function}\n\nOnly the {buffer} best models proceed to the next layer of expansion')
    tested_k_exprs = [start_kernel]
    tested_models = [[GPModel(X, Y, start_kernel).fit(restarts)]]
    methodcaller(utility_function)(tested_models[0][0])
    for d in range(1, depth + 1):
        tested_models.append([]) # tested_models[d] = []
        model_buffer = tested_models[d - 1][:buffer] if len(tested_models[d - 1]) >= buffer else tested_models[d - 1]
        new_k_exprs = [kex for kex in unique(flatten([expand(model.kernel_expression, p_rules) for model in model_buffer])) if kex not in tested_k_exprs]
        tested_models[d] += fit_model_list(X, Y, new_k_exprs, restarts)
        tested_k_exprs += new_k_exprs
        tested_models[d].sort(key = methodcaller(utility_function))
        if verbose: print(f'\tBest depth-{d} models (descending): {", ".join([str(x.kernel_expression) for x in (tested_models[d][:buffer] if len(tested_models[d]) >= buffer else tested_models[d])])}')
    sorted_models = sorted(flatten(tested_models), key = attrgetter('cached_utility_function'))
    best_models = sorted_models[:buffer] if len(sorted_models) >= buffer else sorted_models
    if verbose: print(f'\tBest models overall (descending): {", ".join([str(x.kernel_expression) for x in best_models])}\n')
    return best_models, tested_models, tested_k_exprs
