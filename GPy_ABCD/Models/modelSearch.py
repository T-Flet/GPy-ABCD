import warnings
from operator import methodcaller, attrgetter
from multiprocessing import Pool, cpu_count
from GPy.models import GPRegression

from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.kernelUtil import score_ps, AIC, AICc, BIC
from GPy_ABCD.Util.genericUtil import flatten


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

    # Kwargs passed to optimize_restarts, which passes them to optimize
    #   Check comments in optimize's class AND optimization.get_optimizer for for real list of optimizers
    # TODO: Eventually set robust to True; see description in optimize_restarts method
    def fit(self, restarts = None, verbose = False, robust = False, **kwargs):
        if restarts is None:
            if self.restarts is None: raise ValueError('No restarts value specified')
        else: self.restarts = restarts
        self.model = GPRegression(self.X, self.Y, self.kernel_expression.to_kernel())
        with warnings.catch_warnings(): # Ignore known numerical warnings
            warnings.simplefilter("ignore")
            self.model.optimize_restarts(num_restarts = self.restarts, verbose = verbose, robust = robust, **kwargs)
        return self

    def interpret(self): return fit_ker_to_kex_with_params(self.model.kern, deepcopy(self.kernel_expression)).get_interpretation()

    def predict(self, X, quantiles = (2.5, 97.5), full_cov = False, Y_metadata = None, kern = None, likelihood = None, include_likelihood = True):
        mean, cov = self.model.predict(X, full_cov, Y_metadata, kern, likelihood, include_likelihood)
        qs = self.model.predict_quantiles(X, quantiles, Y_metadata, kern, likelihood)
        return {'mean': mean, 'covariance': cov, 'low_quantile': qs[0], 'high_quantile': qs[1]}



    # Model fit objective criteria & related values:

    def _ll(self): return self.model.log_likelihood()
    def _n(self): return len(self.model.X) # number of data points
    def _k(self): return self.model._size_transformed() # number of estimated parameters, i.e. model degrees of freedom

    def AIC(self):
        self.cached_utility_function = AIC(*score_ps(self.model))
        self.cached_utility_function_type = 'AIC'
        return self.cached_utility_function

    def AICc(self): # Assumes univariate model linear in its parameters and with normally-distributed residuals conditional upon regressors
        self.cached_utility_function = AICc(*score_ps(self.model))
        self.cached_utility_function_type = 'AICc'
        return self.cached_utility_function

    def BIC(self):
        self.cached_utility_function = BIC(*score_ps(self.model))
        self.cached_utility_function_type = 'BIC'
        return self.cached_utility_function


def print_k_list(k_or_model_list):
    return ', '.join([str(m.kernel_expression) for m in k_or_model_list] if isinstance(k_or_model_list[0], GPModel) else [str(k) for k in k_or_model_list])



## Model Sarch

# NOTE: Any code that calls fit_model_list (however many nested levels in), needs to be within a "if __name__ == '__main__':"
#       preamble in order to work on Windows. Note that this includes find_best_model, and hence any call to this project.

def fit_one_model(X, Y, kex, restarts): return GPModel(X, Y, kex).fit(restarts)
def fit_model_list_not_parallel(X, Y, k_exprs, restarts = 5):
    return [fit_one_model(X, Y, kex, restarts) for kex in k_exprs]
def fit_model_list_parallel(X, Y, k_exprs, restarts = 5):
    with Pool() as pool: return pool.starmap_async(fit_one_model, [(X, Y, kex, restarts) for kex in k_exprs], int(len(k_exprs) / cpu_count()) + 1).get()


# start_kernels = make_simple_kexs(['WN']) #for the original ABCD
def explore_model_space(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all, utility_function = 'BIC',
                        restarts = 5, rounds = 2, buffer = 4, dynamic_buffer = True, verbose = False, parallel = True):
    """
    Perform `rounds` rounds of kernel expansion followed by model fit starting from the given `start_kernels` with and
    expanding the best `buffer` of them with `p_rules` production rules

    :param start_kernels: the starting kernels
    :type start_kernels: [KernelExpression]
    :param p_rules: the production rules applied at each expansion
    :type p_rules: [function]
    :param utility_function: Name of utility function: AIC, AICc and BIC available so far (will allow function input in future releases)
    :type utility_function: String
    :param restarts: Number of GPy model-fitting restarts with different parameters
    :type restarts: Int
    :param rounds: Number of rounds of model exploration
    :type rounds: Int
    :param buffer: Number of best fit-models' kernels to expand each round
    :type buffer: Int
    :param dynamic_buffer: If True: buffer is increased by 2 at the beginning and decreased by 1 in the first two and last two rounds
    :type dynamic_buffer: Boolean
    :param verbose: Produce verbose logs
    :type verbose: Boolean
    :param parallel: Perform multiple model fits concurrently on all available processors (vs GPy's own parallel argument, which splits a single fit over multiple processors)
    :type parallel: Boolean

    :rtype: (sorted_models: [GPModel], tested_models: [[GPModel]], tested_k_exprs: [KernelExpression], expanded: [GPModel], not_expanded: [GPModel])
    """
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    start_kexs = make_simple_kexs(start_kernels)

    if verbose: print(f'Testing {rounds} layers of model expansion on {len(X)} datapoints, starting from: {print_k_list(start_kexs)}\nModels are fitted with {restarts} random restarts and scored by {utility_function}')
    fit_model_list = fit_model_list_parallel if parallel else fit_model_list_not_parallel

    tested_models = [sorted(fit_model_list(X, Y, start_kexs, restarts), key = methodcaller(utility_function))]
    sorted_models = not_expanded = sorted(flatten(tested_models), key = methodcaller(utility_function))
    expanded = []
    tested_k_exprs = deepcopy(start_kexs)

    original_buffer = buffer
    if dynamic_buffer: buffer += 2 # Higher for the 1st round
    if verbose: print(f'(All models are listed by descending {utility_function})\n\nBest round-{0} models [{buffer} moving forward]: {print_k_list(not_expanded[:buffer])}')

    sorted_models, tested_models, tested_k_exprs, expanded, not_expanded = model_search_rounds(X, Y,
                   original_buffer, sorted_models, tested_models, tested_k_exprs, expanded, not_expanded, fit_model_list,
                   p_rules, utility_function, restarts, rounds, buffer, dynamic_buffer, verbose)

    if verbose: print(f'\nBest models overall: {print_k_list(sorted_models[:original_buffer])}\n')
    return sorted_models, tested_models, tested_k_exprs, expanded, not_expanded


# This function is split from the above both for tidiness and to allow the possibility of continuing a search if desired
def model_search_rounds(X, Y, original_buffer, sorted_models, tested_models, tested_k_exprs, expanded, not_expanded,
                        fit_model_list, p_rules, utility_function, restarts, rounds, buffer, dynamic_buffer, verbose):
    """
    See explore_model_space description and source code for argument explanation and context

    Note: sorted_models is not actually used but replaced with the new value; present as an argument just for consistency
    """
    for d in range(1, rounds + 1):
        new_k_exprs = [kex for kex in unique(flatten([expand(mod.kernel_expression, p_rules) for mod in not_expanded[:buffer]])) if kex not in tested_k_exprs]
        tested_models.append(sorted(fit_model_list(X, Y, new_k_exprs, restarts), key = methodcaller(utility_function)))  # tested_models[d]

        sorted_models = sorted(flatten(tested_models), key = attrgetter('cached_utility_function')) # Merge-sort would be applicable
        expanded += not_expanded[:buffer]
        not_expanded = diff(sorted_models, expanded) # More efficient than sorting another whole list
        tested_k_exprs += new_k_exprs

        buffer -= 1 if dynamic_buffer and (d <= 2 or d in range(rounds - 1, rounds + 1)) else 0
        if verbose: print(f'Round-{d} models [{buffer} moving forward]:\n\tBest new: {print_k_list(tested_models[d][:original_buffer])}\n\tBest so far: {print_k_list(sorted_models[:original_buffer])}\n\tBest not-already-expanded: {print_k_list(not_expanded[:buffer])}')

    return sorted_models, tested_models, tested_k_exprs, expanded, not_expanded



# TODO:
#   - focus on documenting end-user and generic developer functions etc in sphinx
#   - make the dynamic buffer configurable
#   - make an interactive mode which asks whether to go further, retaining how many etc
#   - allow the model lists in each round to be fit in batches, with interactive request to continue (timed response maybe)
#   - show an updating count of models having been fitted so far in this round; at least by batches
#   - Make the utility_function argument optionally a function taking (ll, n, k) as arguments as in kernelUtil
