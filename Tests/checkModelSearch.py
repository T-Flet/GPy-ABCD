from GPy_ABCD.Models.modelSearch import *
import numpy as np


if __name__ == '__main__':

    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


    X = np.linspace(-10, 10, 101)[:, None]

    Y = np.cos( (X - 5) / 2 )**2 * X * 2 + np.random.randn(101, 1) * 1 #- 100
    # Y = np.concatenate(([0.1 * x for x in X[:50]], np.array([0])[:, None], [2 + 3 * np.sin(x*3) for x in X[51:]])) + np.random.randn(101, 1) * 0.3

    # from Util.kernelUtil import doGPR
    # mod = doGPR(X, Y, LIN() * (PER() + C()), 10)
    # predict_X = np.linspace(10, 15, 50)[:, None]
    # (p_mean, p_var) = mod.predict(predict_X)
    # print(predicted)


    # # Load testing for parallel computations
    # from timeit import timeit
    # def statement():
    #     best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernels=standard_start_kernels, p_rules=production_rules_all,
    #                                                      restarts=2, utility_function='BIC', rounds=2, buffer=3, verbose=True, parallel=True)
    # print(timeit(statement, number = 3))



    # best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernels = [SumKE(['WN'])._initialise()], p_rules = production_rules_all,
    best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all,
                                                     restarts = 2, utility_function = 'BIC', rounds = 2, buffer = 3, verbose = True, parallel = True)

    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    from matplotlib import pyplot as plt
    for bm in best_mods:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()
        print(bm.interpret())


    predict_X = np.linspace(10, 15, 50)[:, None]
    preds = best_mods[0].predict(predict_X)
    print(preds)

    plt.show()


# TODO:
#   Address these testing notes:
# 	- Changepoint and changewindow kernels seem to throw off PER a bit, and additionally allow SE to match most data
#   - Change something in grammar: use change_point_linear, or make the starting change kernels use SE or other
