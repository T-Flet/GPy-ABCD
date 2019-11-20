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


def print_k_list(k_or_model_list):
    return ', '.join([str(m.kernel_expression) for m in k_or_model_list] if isinstance(k_or_model_list[0], GPModel) else [str(k) for k in k_or_model_list])


## Model Sarch

# NOTE: Any code that calls (however many nested levels in) fit_model_list, needs to be within a
#   if __name__ == '__main__':
#   preamble in order to work on Windows. Note that this includes find_best_model, and hence any call to this project.

def fit_one_model(X, Y, kex, restarts): return GPModel(X, Y, kex).fit(restarts)
def fit_model_list(X, Y, k_exprs, restarts = 5):
    with Pool() as pool: return pool.starmap_async(fit_one_model, [(X, Y, kex, restarts) for kex in k_exprs], int(len(k_exprs) / cpu_count()) + 1).get()

# TODO:
#  Decide whether to have a set of initial common models before or instead of WN expansion in order to return something quickly
#   (this is related to the different sets of production rules possibly being applied at different depth levels)
#   Introduce stopping of some branches of model testing if, say, no models spawned from a specific one manage to beat it;
#       none of them should be expanded further; maybe count the expanded terms to do the check and slice the full list?

def find_best_model(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all, restarts = 5,
                    utility_function = 'BIC', rounds = 2, buffer = 5, verbose = False):
    if verbose: print(f'Testing {rounds} layers of model expansion starting from: {print_k_list(start_kernels)}\nModels are fitted with {restarts} random restarts and scored by {utility_function}\n\nOnly the {buffer} best not-already-expanded models proceed to each subsequent layer of expansion')

    tested_models = [sorted(fit_model_list(X, Y, start_kernels, restarts), key = methodcaller(utility_function))]
    sorted_models = not_expanded = tested_models[0]
    expanded = []
    tested_k_exprs = start_kernels
    if verbose: print(f'(All models are listed in descending order)\n\nBest round-{0} models: {print_k_list(not_expanded[:buffer])}')

    for d in range(1, rounds + 1):
        new_k_exprs = [kex for kex in unique(flatten([expand(mod.kernel_expression, p_rules) for mod in not_expanded[:buffer]])) if kex not in tested_k_exprs]
        tested_models.append(sorted(fit_model_list(X, Y, new_k_exprs, restarts), key = methodcaller(utility_function))) # tested_models[d]
        sorted_models = sorted(flatten(tested_models), key = attrgetter('cached_utility_function')) # Merge-sort would be applicable
        expanded += not_expanded[:buffer]
        not_expanded = lists_of_unhashables__diff(sorted_models, expanded) # More efficient than sorting another whole list
        tested_k_exprs += new_k_exprs
        if verbose: print(f'Round-{d} models:\n\tBest new: {print_k_list(tested_models[d][:buffer])}\n\tBest so far: {print_k_list(sorted_models[:buffer])}\n\tBest not-already-expanded: {print_k_list(not_expanded[:buffer])}')

    if verbose: print(f'\nBest models overall: {print_k_list(sorted_models[:buffer])}\n')
    return sorted_models[:buffer], tested_models, tested_k_exprs
