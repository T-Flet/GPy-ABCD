import numpy as np

from GPy_ABCD.Models.modelSearch import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from testConsistency import save_one_run
from synthetic_datasets import dataset, X, Y, correct_k, kernel


if __name__ == '__main__':

    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

    # from GPy_ABCD.Util.kernelUtil import doGPR
    # mod = doGPR(X, Y, kernel, 10)
    # predict_X = np.linspace(10, 15, 50)[:, None]
    # (p_mean, p_var) = mod.predict(predict_X)
    # print(predicted)


    # # Load testing for parallel computations
    # from timeit import timeit
    # def statement():
    #     best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels=standard_start_kernels, p_rules=production_rules_all,
    #                                                      restarts=2, utility_function='BIC', rounds=2, buffer=3, dynamic_buffer = True, verbose=True, parallel=True)
    # print(timeit(statement, number = 3))


    # best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels = ['WN'], p_rules = production_rules_all,
    # best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels = test_start_kernels, p_rules = production_rules_all,
    # best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels = extended_start_kernels, p_rules = production_rules_all,
    best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all,
                                 restarts = 3, utility_function = 'BIC', rounds = 2, buffer = 2, dynamic_buffer = True, verbose = True, parallel = True)


    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    print()

    from matplotlib import pyplot as plt
    for bm in best_mods[:3]:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()
        print(bm.interpret())


    # predict_X = np.linspace(10, 15, 50)[:, None]
    # preds = best_mods[0].predict(predict_X)
    # print(preds)


    plt.show()


    save_one_run(dataset, correct_k, best_mods, all_mods, all_exprs)


# TODO:
#   Address these testing notes:
# 	- Changepoint and changewindow kernels seem to throw off PER a bit, and additionally allow SE to match most data
#   - Change something in grammar: use change_point_linear, or make the starting change kernels use SE or other
