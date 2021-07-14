import numpy as np
from copy import deepcopy
from operator import attrgetter
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Dict, Tuple, Union

from GPy_ABCD.KernelExpressions.base import KernelExpression
from GPy_ABCD.KernelExpansion.grammar import start_kernels, production_rules, make_simple_kexs, expand
from GPy_ABCD.Util.modelUtil import *
from GPy_ABCD.Util.genericUtil import flatten, diff, unique


# TODO:
#   - group input parameters into dictionaries?
#   - make utility functions take in the kernel expression (or even full model) so that further criteria may be applied (e.g. presence of specific kernels etc)?
#   - focus on documenting end-user and generic developer functions etc in sphinx
#   - make the dynamic buffer configurable, or even allow inputting a list of numbers of models to keep per round
#   - make an interactive/interruptable mode which asks whether to go further, retaining how many etc
#   - allow the model lists in each round to be fit in batches, with interactive request to continue (timed response maybe)
#   - show a live count of models fitted so far in each round (probably by batches)


def print_k_list(k_or_model_list):
    return ', '.join([str(m.kernel_expression) for m in k_or_model_list] if isinstance(k_or_model_list[0], GPModel) else [str(k) for k in k_or_model_list])


def fit_one_model(X, Y, kex, restarts, optimiser = GPy_optimisers[0]): return GPModel(X, Y, kex).fit(restarts, optimiser)

def fit_mods_not_parallel(X, Y, k_exprs, restarts = 5, optimiser = GPy_optimisers[0]):
    '''Fit models from a list of kernels to the same data NOT in parallel'''
    return [fit_one_model(X, Y, kex, restarts, optimiser) for kex in k_exprs]
def fit_mods_parallel_processes(X, Y, k_exprs, restarts = 5, optimiser = GPy_optimisers[0]):
    '''Concurrently fit models (from a list of kernels) on the same data in as many processes as there are available processor cores using `multiprocessing`'s `Pool`
    (this function does NOT use GPy's own `parallel` argument, which parallelises within single fits rather than across multiple ones)

    NOTE: Any code calling this function (however many nested levels in) should be within a "if __name__ == '__main__':" preamble for full OS-agnostic use.
    '''
    with Pool() as pool: return pool.starmap_async(fit_one_model, [(X, Y, kex, restarts, optimiser) for kex in k_exprs], int(len(k_exprs) / cpu_count()) + 1).get()


def explore_model_space(X, Y,
                        start_kernels: List[Union[str, KernelExpression]] = start_kernels['Default'], p_rules: List[Callable] = production_rules['Default'], utility_function: Callable = BIC,
                        rounds: int = 2, beam: Union[int, List[int]] = [3, 2, 1], restarts: int = 5,
                        model_list_fitter: Callable = fit_mods_parallel_processes, optimiser: str = GPy_optimisers[0],
                        verbose: bool = True) -> Tuple[List[GPModel], List[List[GPModel]], List[KernelExpression], List[GPModel], List[GPModel]]:
    '''Perform `rounds` rounds of kernel expansion followed by model fit starting from the given `start_kernels` with and expanding the best `buffer` of them with `p_rules` production rules

    NOTE: if the default `model_list_fitter` argument `fit_mods_parallel_processes` is used the function should be called from within a :code:`if __name__ == '__main__':` for full OS-agnostic use.

    :param start_kernels: the 0th-round starting kernels
    :type start_kernels: List[Union[str, KernelExpression]]
    :param p_rules: the production rules applied at each expansion
    :type p_rules: List[Callable]
    :param utility_function: model-scoring utility function: inputs are log-likelihood (ll), number of data points (n) and number of estimated parameters (k); AIC, AICc and BIC functions exported; arbitrary ones accepted
    :type utility_function: Callable
    :param rounds: number of rounds of model exploration
    :type rounds: Int
    :param beam: number of best fit-models' kernels to expand each round, either an integer or a list of integers; in the latter case, if its length is less than rounds then the last value will be repeated until required
    :type beam: Union[Int, List[Int]]
    :param restarts: number of GPy model-fitting restarts with different parameters
    :type restarts: Int
    :param model_list_fitter: function handling the fitting of a list of kernels to the same data; this is where parallelisation implementation can be changed
    :type model_list_fitter: Callable
    :param optimiser: identifying string for the model optimiser function; GPy 1.9.9 optimiser strings (GPy > paramz > optimization > optimization.py): 'lbfgsb', 'org-bfgs', 'fmin_tnc', 'scg', 'simplex', 'adadelta', 'rprop', 'adam'
    :type optimiser: str
    :param verbose: produce verbose logs
    :type verbose: Boolean

    :rtype: (sorted_models: [GPModel], tested_models: [[GPModel]], tested_k_exprs: [KernelExpression], expanded: [GPModel], not_expanded: [GPModel])
    '''
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]

    start_kexs = make_simple_kexs(start_kernels)

    if verbose: print(f'Testing {rounds} layers of model expansion on {len(X)} datapoints, starting from: {print_k_list(start_kexs)}\nModels are fitted with {restarts} random restarts (with {optimiser} optimiser) and scored by {utility_function.__name__}')
    def score(m): return m.compute_utility(utility_function)

    tested_models = [sorted(model_list_fitter(X, Y, start_kexs, restarts, optimiser), key = score)]
    sorted_models = not_expanded = sorted(flatten(tested_models), key = score)
    expanded = []
    tested_k_exprs = deepcopy(start_kexs)

    # Handle beam argument: ensure it is a list of integers of length rounds
    if isinstance(beam, list) and all(isinstance(x, int) for x in beam):
        beam = beam + ([beam[-1]] * more) if (more := rounds - len(beam)) > 0 else beam[:rounds]
    elif isinstance(beam, int): beam = [beam] * rounds
    else: raise TypeError(f'The given beam argument ({beam}) is neither an integer nor a list of integers')

    if verbose: print(f'(All models are listed by descending {utility_function.__name__})\n\nBest round-{0} models [{len(tested_models[0])} new; {beam[0]} moving forward]: {print_k_list(not_expanded[:beam[0]])}')

    sorted_models, tested_models, tested_k_exprs, expanded, not_expanded = model_search_rounds(X, Y,
                        sorted_models, tested_models, tested_k_exprs, expanded, not_expanded,
                        model_list_fitter, p_rules, utility_function, rounds, beam, restarts, optimiser, verbose)

    if verbose: print(f'\nBest models overall: {print_k_list(sorted_models[:beam[0]])}\n')
    return sorted_models, tested_models, tested_k_exprs, expanded, not_expanded


# This function is split from the above both for tidiness and to allow the possibility of continuing a search if desired
def model_search_rounds(X, Y, sorted_models, tested_models, tested_k_exprs, expanded, not_expanded,
                        model_list_fitter, p_rules, utility_function, rounds, beam, restarts, optimiser, verbose):
    '''
    This function can be used to continue a previous search: the 5 arguments after Y are the outputs of explore_model_space.
    See explore_model_space description and definition for argument explanation and context.

    Note: sorted_models is not actually used but replaced with the new value; present as an argument just for consistency
    '''
    def score(m): return m.compute_utility(utility_function)

    for r, b in zip(range(1, rounds + 1), beam):
        new_k_exprs = [kex for kex in unique(flatten([expand(mod.kernel_expression, p_rules) for mod in not_expanded[:b]])) if kex not in tested_k_exprs]
        tested_models.append(sorted(model_list_fitter(X, Y, new_k_exprs, restarts, optimiser), key = score))  # tested_models[d]

        sorted_models = sorted(flatten(tested_models), key = attrgetter('cached_utility_function')) # Merge-sort would be applicable
        expanded += not_expanded[:b]
        not_expanded = diff(sorted_models, expanded) # More efficient than sorting another whole list
        tested_k_exprs += new_k_exprs

        if verbose: print(f'Round-{r} models [{len(tested_models[r])} new; {b} moving forward]:\n\tBest new: {print_k_list(tested_models[r][:beam[0]])}\n\tBest so far: {print_k_list(sorted_models[:beam[0]])}\n\tBest not-already-expanded: {print_k_list(not_expanded[:b])}')

    return sorted_models, tested_models, tested_k_exprs, expanded, not_expanded


