import numpy as np
from operator import methodcaller
from GPy.models import GPRegression
from kernelExpression import *
from grammar import *


class GPModel():
    def __init__(self, X, Y, kernel_expression = SumKE(['WN'])._initialise()):
        self.X = X
        self.Y = Y
        self.kernel_expression = kernel_expression
        self.restarts = None
        self.model = None
        self.cached_BIC = None

    def fit(self, restarts = None):
        if restarts is None:
            if self.restarts is None: raise ValueError('No restarts value specified')
        else: self.restarts = restarts
        self.model = GPRegression(self.X, self.Y, self.kernel_expression.to_kernel())
        self.model.optimize_restarts(num_restarts = self.restarts)
        return self

    def BIC(self): # model.X is the number of points, and model._size_transformed() is the number of optimisation parameters
        self.cached_BIC = -2 * self.model.log_likelihood() + self.model._size_transformed() * np.log(len(self.model.X))
        return self.cached_BIC


# Model Testing functions

def find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules, restarts = 5, utility_function = 'BIC', depth = 1, buffer = 5):
    tested_k_exprs = [start_kernel]
    tested_models = [[GPModel(X, Y, start_kernel).fit(restarts)]]
    for d in range(1, depth + 1):
        tested_models[d - 1].sort(key = methodcaller('BIC'))
        tested_models.append([]) # tested_models[d] = []
        for i, model in enumerate(tested_models[d - 1]):
            if i >= buffer: break
            for kex in expand(model.kernel_expression, p_rules.values()):
                if kex not in tested_k_exprs:
                    mod = GPModel(X, Y, kex).fit(restarts)
                    tested_k_exprs.append(kex)
                    tested_models[d].append(mod)
    tested_models[depth].sort(key=methodcaller('BIC'))
    return sorted(flatten(tested_models), key=methodcaller('BIC'))[:5], tested_models, tested_k_exprs







import numpy as np
from Util.util import doGPR

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


X = np.linspace(-10, 10, 101)[:, None]

Y = np.cos( (X - 5) / 2 )**2 * 7 + np.random.randn(101, 1) * 1 #- 100

# doGPR(X, Y, PER + C, 10)


best_mods, all_exprs, all_mods = find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules,
                                                 restarts = 5, utility_function = 'BIC', depth = 2, buffer = 5)

from matplotlib import pyplot as plt
for bm in best_mods:
    print(bm.kernel_expression)
    print(bm.model.kern)
    print(bm.model.log_likelihood())
    print(bm.cached_BIC)
    bm.model.plot()

plt.show()
