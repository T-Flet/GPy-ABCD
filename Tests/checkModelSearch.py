import numpy as np

from GPy_ABCD.Models.modelSearch import *
from GPy_ABCD.Util.modelUtil import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from Tests.testConsistency import save_one_run
from Tests.synthetic_datasets import dataset, X, Y, correct_k, kernel


if __name__ == '__main__':
    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

    ## Forced data (using X and Y from synthetic_datasets otherwise
    # X = np.linspace(-10, 10, 101)[:, None]
    # Y = np.cos((X - 5) / 2) ** 2 * X * 2 + np.random.randn(101, 1)


    ## Single fit
    # mod = fit_kex(X, Y, correct_k, 10)#, optimiser = GPy_optimisers[4])
    # model_printout(mod)
    # # predict_X = np.linspace(10, 15, 50)[:, None]
    # # pred = mod.predict(predict_X)
    # # print(pred['mean'])


    ## Load testing for parallel computations
    # from timeit import timeit
    # def statement():
    #     best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels=standard_start_kernels, p_rules=production_rules_all,
    #                                                      utility_function=BIC, restarts=2, rounds=2, buffer=3, dynamic_buffer = True, verbose=True, parallel=True)
    # print(timeit(statement, number = 3))


    # test_start_kernels = make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
    #                                       [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
    #                                       both_changes('LIN')) # To catch a possible changepoint or changewindow with simple enough shapes


    ## Model search
    best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y,
                                start_kernels = start_kernels['Default'], p_rules = production_rules['Default'],
                                utility_function = BIC, restarts = 4, rounds = 1, buffer = 3, dynamic_buffer = False,
                                verbose = True, parallel = True, optimiser = GPy_optimisers[0])


    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    print()


    from matplotlib import pyplot as plt
    for bm in best_mods[:3]:
        model_printout(bm)
        # print('2022:', bm.predict(np.array([2022])[:, None]))


    # predict_X = np.linspace(10, 15, 10)[:, None]
    # preds = best_mods[0].predict(predict_X)
    # print(preds)


    plt.show()


    # save_one_run(dataset, correct_k, best_mods, all_mods, all_exprs)


# TODO:
#   Address these testing notes:
# 	- Changepoint and changewindow kernels seem to throw off PER a bit, and additionally allow SE to match most data
#   - Change something in grammar: use change_point_linear, or make the starting change kernels use SE or other
