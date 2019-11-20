from modelSearch import *
import numpy as np
import scipy.io as sio


if __name__ == '__main__':

    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

    data = sio.loadmat('..\\Data\\02-solar.mat')
    # print(data.keys())

    X = data['X']

    Y = data['y']



    best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernel = SumKE(['WN'])._initialise(), p_rules = production_rules_all,
                                                     restarts = 2, utility_function = 'BIC', depth = 1, buffer = 5, verbose= True)

    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]))

    from matplotlib import pyplot as plt
    for bm in best_mods:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()

    plt.show()
