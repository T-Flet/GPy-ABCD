import numpy as np
import scipy.io as sio

from GPy_ABCD.Models.modelSearch import *
from testConsistency import save_one_run


if __name__ == '__main__':

    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger
    datasets = ['01-airline', '02-solar', '03-mauna', '04-wheat', '05-temperature', '06-internet', '07-call-centre', '08-radio', '09-gas-production', '10-sulphuric', '11-unemployment', '12-births', '13-wages']
    dataset_name = datasets[1-1]

    data = sio.loadmat(f'./Data/{dataset_name}.mat')
    # print(data.keys())

    X = data['X']
    Y = data['y']

    sorted_models, tested_models, tested_k_exprs, expanded, not_expanded = explore_model_space(X, Y, start_kernels = standard_start_kernels, p_rules = production_rules_all,
                                                     restarts = 3, utility_function = 'BIC', rounds = 2, buffer = 2, dynamic_buffer = True, verbose = True, parallel = True)

    for mod_depth in tested_models: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    from matplotlib import pyplot as plt
    for bm in sorted_models[:3]:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()
        print(bm.interpret())

    plt.show()

    save_one_run(dataset_name, 'UNKNOWN', sorted_models, tested_models, tested_k_exprs)

