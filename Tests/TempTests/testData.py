import numpy as np
import scipy.io as sio

from GPy_ABCD.Models.modelSearch import *


# TODO:
#   Solar data: the changewindow is not appropriately captured; need to investigate how to best address this;
#   maybe add the second location parameter to CW

if __name__ == '__main__':

    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

    data = sio.loadmat('./Data/02-solar.mat')
    # print(data.keys())

    X = data['X']

    Y = data['y']



    best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all,
                                                     restarts = 2, utility_function = 'BIC', rounds = 2, buffer = 2, dynamic_buffer = True, verbose = True, parallel = True)

    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    from matplotlib import pyplot as plt
    for bm in best_mods[:3]:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()
        print(bm.interpret())

    plt.show()
