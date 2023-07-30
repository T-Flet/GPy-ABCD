import numpy as np
from GPy_ABCD import *


if __name__ == '__main__':
    # Example data
    X = np.linspace(-10, 10, 101)[:, None]
    Y = np.cos((X - 5) / 2) ** 2 * X * 2 + np.random.randn(101, 1)

    # Main function call with default arguments
    best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y,
        start_kernels = start_kernels['Default'], p_rules = production_rules['Default'],
        utility_function = BIC, rounds = 2, beam = [3, 2, 1], restarts = 5,
        model_list_fitter = fit_mods_parallel_processes, optimiser = GPy_optimisers[0],
        verbose = True)

    print('\nFull lists of models by round:')
    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    print('\n\nTop-3 models\' details:')
    for bm in best_mods[:3]:
        model_printout(bm, plotly = False) # See the definition of this convenience function for examples of model details' extraction
        print('Prediction at X = 11:', bm.predict(np.array([11])[:, None]), '\n')

    from matplotlib import pyplot as plt
    plt.show() # Not required for plotly = True above


