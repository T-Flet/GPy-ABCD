'''This file is an old, reduced, Python-3.8-compatible version of the one in https://github.com/T-Flet/Python-Generic-Util'''
import time
from pandas import DataFrame, concat
from numba import njit, vectorize

from typing import Callable, Dict, List, Tuple


def timethis(f: Callable, n = 2, *args, **kwargs):
    '''Return n execution times for the given function with given arguments'''
    ts = []
    for i in range(n):
        ts.append(time.perf_counter())
        f(*args, **kwargs)
        ts[-1] = time.perf_counter() - ts[-1]
    return ts


def compare_implementations(fs_with_shared_args: Dict[str, Callable], n = 200, wait = 1, verbose = True,
                            fs_with_own_args: Dict[str, Tuple[Callable, List, Dict]] = None, args: List = None, kwargs: Dict = None):
    '''Benchmark multiple implementations of the same function called n times each with the same args and kwargs, with a break between functions.
        fs_with_own_args is meant for additional functions taking different *args and **kwargs.
        Recommended output view if not verbose: print(table.to_markdown(index = False)).'''
    assert n >= 3
    table = []
    for name, f in fs_with_shared_args.items():
        time.sleep(wait)
        if args:
            if kwargs: times = timethis(f, n, *args, **kwargs)
            else: times = timethis(f, n, *args)
        elif kwargs: times = timethis(f, n, **kwargs)
        else: times = timethis(f, n)
        table.append([name, sum(times) / len(times), sum(times[1:]) / (len(times)-1), times[0], times[1], times[2]])
        if verbose: print(f'Benchmarked {name} - mean {table[-1][1]} and mean excluding 1st run {table[-1][2]}')

    if fs_with_own_args:
        for name, f_a_k in fs_with_own_args.items():
            f, args, kwargs = f_a_k
            time.sleep(wait)
            if args:
                if kwargs: times = timethis(f, n, *args, **kwargs)
                else: times = timethis(f, n, *args)
            elif kwargs: times = timethis(f, n, **kwargs)
            else: times = timethis(f, n)
            table.append([name, sum(times) / len(times), sum(times[1:]) / (len(times)-1), times[0], times[1], times[2]])
            if verbose: print(f'Benchmarked {name} - mean {table[-1][1]} and mean excluding 1st run {table[-1][2]}')

    table = sorted(table, key = lambda row: row[1])

    last, last1 = table[0][1], table[0][2]
    table = [(name, m0, m1, m0 / table[0][1], m1 / table[0][2], next_mean_ratio, next_mean_ratio1, t0, t1, t2)
             for name, m0, m1, t0, t1, t2 in table
             if (next_mean_ratio := m0 / last) if (next_mean_ratio1 := m1 / last1)
             if (last := m0) if (last1 := m1)]

    df = DataFrame(table, columns = ['f', 'mean', 'mean excl. 1st', 'best mean ratio', 'best mean1 ratio', 'next mean ratio', 'next mean1 ratio', 't0', 't1', 't2'])
    if verbose: print('\n', df.to_markdown(index = False), sep = '')
    return df


def numba_comparisons(f: Callable = None, numba_signature = None, separate_numba_signatures = None,
                      f_scalar: Callable = None, numba_signatures_scalar = None, separate_numba_signaturess_scalar = None, parallel = False,
                      fs_with_own_args: Dict[str, Tuple[Callable, List, Dict]] = None, n = 200, wait = 1, prefix = None, verbose = True,
                      args: List = None, kwargs: dict = None):
    '''Combine available ingredients to benchmark possible numba versions of the given function.
        numba_signature and parallel are only used if f is given.
        numba_signatures_scalar is only used if f_scalar is given.
        separate_numba_signatures and separate_numba_signaturess_scalar are, respectively, a list and a list of lists of
        additional signatures to be tried in addition to, respectively, numba_signature and numba_signatures_scalar.'''
    f_dict = dict()
    if f:
        f_dict['Base'], f_dict['Lazy'] = f, njit(f)
        if parallel: f_dict['Parallel'] = njit(parallel = True)(f)
        if numba_signature:
            f_dict['Eager'] = njit(numba_signature)(f)
            if parallel: f_dict['Parallel Eager'] = njit(numba_signature, parallel = True)(f)
        if separate_numba_signatures:
            for i, sig in enumerate(separate_numba_signatures):
                f_dict[f'Eager{i+2}'] = njit(sig)(f)
                if parallel: f_dict[f'Parallel Eager{i+2}'] = njit(sig, parallel = True)(f)
    if f_scalar:
        f_dict['Vec Lazy'] = vectorize(f_scalar)
        if numba_signatures_scalar: f_dict['Vec Eager'] = vectorize(numba_signatures_scalar)(f_scalar)
        if separate_numba_signaturess_scalar:
            for i, sigs in enumerate(separate_numba_signaturess_scalar): f_dict[f'Vec Eager{i+2}'] = vectorize(sigs)(f_scalar)
    if prefix: f_dict = {(prefix + k): v for k, v in f_dict.items()}
    return compare_implementations(f_dict, n, wait, verbose, fs_with_own_args, args, kwargs)


def merge_bench_tables(*tables, verbose = True):
    table = concat(tables).sort_values(by = 'mean').reset_index(drop = True)
    table['best mean ratio'] = table['mean'] / table['mean'].values[0]
    table['best mean1 ratio'] = table['mean excl. 1st'] / table['mean excl. 1st'].values[0]
    table['next mean ratio'] = table['mean'] / table['mean'].shift()
    table['next mean1 ratio'] = table['mean excl. 1st'] / table['mean excl. 1st'].shift()
    table.loc[0, 'next mean ratio'], table.loc[0, 'next mean1 ratio'] = 1, 1
    if verbose: print('\n', table.to_markdown(index = False), sep = '')
    return table


