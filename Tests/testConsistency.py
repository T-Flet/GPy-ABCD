import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from GPy_ABCD.Models.modelSearch import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from synthetic_datasets import *


def get_model_round(bm, all_mods):
    for i in range(len(all_mods)):
        if bm.kernel_expression in [m.kernel_expression for m in all_mods[i]]: return i
    raise ValueError('Somehow this model is not among the tested models; equality checking failure?')


def one_run_statistics(best_mods, all_mods, all_exprs, top_n):
    res = defaultdict(list)
    for m in best_mods[:top_n]:
        res['kex'].append(m.kernel_expression)
        res['utility'].append(m.cached_utility_function)
        res['round'].append(get_model_round(m, all_mods))
    return res


def get_and_save_stats(n_runs_stats, dataset_name = dataset, expected_best = correct_k):
    n_iterations = len(n_runs_stats)
    final_stats = defaultdict(list)
    for i in range(n_iterations):
        final_stats['correct'].append(str(expected_best))
        final_stats['correct_1st'].append(sum([expected_best == n_runs_stats[j]['kex'][0] for j in range(n_iterations)]) / n_iterations)
        final_stats['correct_in_2'].append(sum([expected_best in n_runs_stats[j]['kex'][:1] for j in range(n_iterations)]) / n_iterations)
        final_stats['correct_in_3'].append(sum([expected_best in n_runs_stats[j]['kex'][:2] for j in range(n_iterations)]) / n_iterations)

        final_stats['1st'].append(str(n_runs_stats[i]['kex'][0]))
        final_stats['1st_round'].append(str(n_runs_stats[i]['round'][0]))
        final_stats['1st_utility'].append(str(n_runs_stats[i]['utility'][0]))
        final_stats['1st_ratio'].append(sum([k == n_runs_stats[i]['kex'][0] for k in [n_runs_stats[j]['kex'][0] for j in range(n_iterations)]]) / n_iterations)
        final_stats['1st_in_2'].append(sum([n_runs_stats[i]['kex'][0] in n_runs_stats[j]['kex'][:1] for j in range(n_iterations)]) / n_iterations)
        final_stats['1st_in_3'].append(sum([n_runs_stats[i]['kex'][0] in n_runs_stats[j]['kex'][:2] for j in range(n_iterations)]) / n_iterations)

        final_stats['2nd'].append(str(n_runs_stats[i]['kex'][1]))
        final_stats['2nd_round'].append(str(n_runs_stats[i]['round'][1]))
        final_stats['2nd_utility'].append(str(n_runs_stats[i]['utility'][1]))
        final_stats['2nd_ratio'].append(sum([k == n_runs_stats[i]['kex'][1] for k in [n_runs_stats[j]['kex'][1] for j in range(n_iterations)]]) / n_iterations)
        final_stats['2nd_in_3'].append(sum([n_runs_stats[i]['kex'][1] in n_runs_stats[j]['kex'][:2] for j in range(n_iterations)]) / n_iterations)

    final_stats = pd.DataFrame(final_stats)
    final_stats.to_csv(f'./Stats/{dataset_name}_{n_iterations}_runs_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    return final_stats


def save_one_run(dataset_name, expected_best, best_mods, all_mods, all_exprs):
    return get_and_save_stats([one_run_statistics(best_mods, all_mods, all_exprs, 5)], dataset_name, expected_best)



if __name__ == '__main__':
    # np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

    n_iterations = 3
    n_runs_stats = []
    for i in range(n_iterations):
        best_mods, all_mods, all_exprs = find_best_model(X, Y, start_kernels=standard_start_kernels, p_rules=production_rules_all,
                                                         restarts=3, utility_function='BIC', rounds=2, buffer=2,
                                                         dynamic_buffer=True, verbose=False, parallel=True)
        n_runs_stats.append(one_run_statistics(best_mods, all_mods, all_exprs, 5))
        print(f'{i+1} runs done')

    get_and_save_stats(n_runs_stats)