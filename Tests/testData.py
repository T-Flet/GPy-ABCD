import pickle
import pprint
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from datetime import datetime

from GPy_ABCD.Models.modelSearch import *
from GPy_ABCD import config as global_flags
from testConsistency import save_one_run


# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger
datasets = ['01-airline', '02-solar', '03-mauna', '04-wheat', '05-temperature', '06-internet', '07-call-centre', '08-radio', '09-gas-production', '10-sulphuric', '11-unemployment', '12-births', '13-wages']
    # Only 1, 2, 10 and 11 have published analyses, and their identified formulae are (deciphered from component descriptions):
    #   1: LIN + PER * LIN + SE + WN * LIN
    #       Default Rules: (PER + C + LIN * (PER + C)) * (PER + PER + WN) * (C + LIN)
    #           or LIN + PER * LIN * (C + WN) * (PER + PER + C), LIN + PER * (C + WN) * (PER + C) * (PER + LIN)
    #   2: C + CW_1643_1716(PER + SE + RQ + WN * LIN + WN * LIN, C + WN)
    #       Default Rules: C + (PER + C) * (PER + PER + PER + WN)
    #  10: PER + SE + CP_64(PER + WN, CW_69_77(SE, SE) + CP_90(C + WN, WN))
    #       Default Rules: (PER + C) * (PER + LIN + PER * LIN)
    #  11: SE + PER + SE + SE + WN
    #       Default Rules: (PER + PER + PER + C) * (C + LIN)


if __name__ == '__main__':
    retrieve_instead = False

    datasets_to_test = [1, 2]#, 10, 11]

    def run_for_dataset_number(dataset_id):
        dataset_name = datasets[dataset_id - 1]
        data = sio.loadmat(f'./Data/{dataset_name}.mat')
        # print(data.keys())

        X = data['X']
        Y = data['y']

        args_to_save = {'start_kernels': start_kernels['Default'], 'p_rules': production_rules['Default'], 'utility_function': BIC,
             'rounds': 5, 'beam': 2, 'restarts': 10, 'model_list_fitter': fit_mods_parallel_processes, 'optimiser': GPy_optimisers[0], 'verbose': True}
        best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y, **args_to_save)

        # for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')
        #
        # print()
        #
        # from matplotlib import pyplot as plt
        # for bm in best_mods[:3]: model_printout(bm)
        # plt.show()

        with open(f'./Pickles/{dataset_name}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}', 'wb') as f:
            pickle.dump({'dataset_name': dataset_name, 'best_mods': best_mods[:10],
                            'str_of_args': pprint.pformat(args_to_save, width = 40, compact = True),
                            'global_flags': {
                                '__INCLUDE_SE_KERNEL': global_flags.__INCLUDE_SE_KERNEL,
                                '__USE_LIN_KERNEL_HORIZONTAL_OFFSET': global_flags.__USE_LIN_KERNEL_HORIZONTAL_OFFSET,
                                '__USE_NON_PURELY_PERIODIC_PER_KERNEL': global_flags.__USE_NON_PURELY_PERIODIC_PER_KERNEL,
                                '__FIX_SIGMOIDAL_KERNELS_SLOPE': global_flags.__FIX_SIGMOIDAL_KERNELS_SLOPE,
                                '__USE_INDEPENDENT_SIDES_CHANGEWINDOW_KERNEL': global_flags.__USE_INDEPENDENT_SIDES_CHANGEWINDOW_KERNEL
            } }, f)

        # save_one_run(dataset_name, 'UNKNOWN', best_mods, all_mods, all_exprs)


    ## ACTUAL EXECUTION ##

    if not retrieve_instead:
        for id in datasets_to_test: run_for_dataset_number(id)
    else:
        file_names = ['01-airline_19-12-2020_19-22-21', '02-solar_19-12-2020_22-27-53', '10-sulphuric_20-12-2020_04-01-08', '11-unemployment_20-12-2020_12-04-19']
        with open(f'./Pickles/{file_names[0]}', 'rb') as f: IMPORTED = pickle.load(f)
        # print(IMPORTED)

        for bm in IMPORTED['best_mods'][:3]: model_printout(bm)
        plt.show()


