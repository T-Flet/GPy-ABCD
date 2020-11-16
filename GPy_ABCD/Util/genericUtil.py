from itertools import chain
from functools import reduce
from collections import defaultdict
import operator as op

from typing import TypeVar, Callable, Union, List, Dict, Iterable, Iterator, Generator, Any, Tuple, Set, Generic, Mapping
_a = TypeVar('_a')
_b = TypeVar('_b')



## Higher-Order Functions

def foldq(f: Callable[[_b, _a], _b], g: Callable[[_b, _a, List[_a]], List[_a]], c: Callable[[_a], bool], xs: List[_a], acc: _b) -> Tuple[_b, List[_a]]:
    r'''
    Fold-like higher-order function where xs is traversed by consumption conditional on c and remaining xs are updated by g (therefore consumption order is not known a priori):
      - the first/next item to be ingested is the first in the remaining xs to fulfil condition c
      - at every x ingestion the item is removed from (a copy of) xs and all the remaining ones are potentially modified by function g
      - this function always returns a tuple of (acc, remaining_xs), unlike the stricter foldq_, which raises an exception for leftover xs
    Note: fold(f, xs, acc) == foldq(f, lambda acc, x, xs: xs, lambda x: True, xs, acc)
    Suitable names: consumption_fold, condition_update_fold, cu_fold, q_fold, qfold or foldq
    :param f: 'Traditional' fold function :: acc -> x -> acc
    :param g: 'Update' function for all remaining xs at every iteration :: acc -> x -> xs -> xs
    :param c: 'Condition' function to select the next x (first which satisfies it) :: x -> Bool
    :param xs: Structure to consume
    :param acc: Starting value for the accumulator
    :returns: (acc, remaining_xs)
    '''
    xs = list(xs) # Copy xs in order not to modify the actual input
    def full_step(acc, xs): # Alternative implementation: move function content inside the while and use a 'broke' flag to trigger a continue before the raise
        for i in range(len(xs)):
            x = xs[i]
            if c(x):
                del xs[i]
                return f(acc, x), g(acc, x, xs)
        return None
    while xs:
        if (res := full_step(acc, xs)): acc, xs = res
        else: break
    return acc, xs

def foldq_(f: Callable[[_b, _a], _b], g: Callable[[_b, _a, List[_a]], List[_a]], c: Callable[[_a], bool], xs: List[_a], acc: _b) -> _b:
    r'''Stricter version of foldq (see its description for details); only returns the accumulator and raises an exception on leftover xs
    :raises ValueError on leftover xs'''
    acc, xs = foldq(f, g, c, xs, acc)
    if xs: raise ValueError('No suitable next element found for given condition while elements remain')
    else: return acc


def partition(p: Callable[[_a], bool], xs: Iterable[_a]) -> Tuple[Iterable[_a], Iterable[_a]]:
    '''Haskell's partition function: partition p xs == (filter p xs, filter (not . p) xs)'''
    acc = ([],[])
    for x in xs: acc[not p(x)].append(x)
    return acc


def group_by(f: Callable[[_a], _b], xs: Iterable[_a]) -> Dict[_b, List[_a]]: # op.itemgetter(x) is a good match
    '''Generalisation of partition to any-output key-function; NOT Haskell's groupBy function'''
    acc = defaultdict(list)
    for x in xs: acc[f(x)].append(x)
    return acc


# Call functions on transformed inputs: by generic function, method or attribute (the first two with optional *args and **kwargs)
def on(f: Callable, xs: Iterable[_a], g: Callable, *args, **kwargs): return f(*[g(x, *args, **kwargs) for x in xs]) # E.g. on(op.gt, (a, b), len); op.itemgetter(x) is a good match
def on_m(f: Callable, xs: Iterable[_a], m: str, *args, **kwargs): return f(*[getattr(x, m)(*args, **kwargs) for x in xs]) # E.g. on_m(op.gt, [a, b], 'count', 'hello')
def on_a(f: Callable, xs: Iterable[_a], a: str): return f(*[getattr(x, a) for x in xs]) # E.g. on_a(op.eq, [a, b], '__class__')


def first(c: Callable[[_a], bool], xs: Iterable[_a], default: _a = None) -> _a: return next((x for x in xs if c(x)), default)



## Iterable-Focussed Functions
# Note: versions of functions with the '_h' suffix only work and are optimal for collections of hashable elements

def topological_sort(nodes_incoming_edges_tuples: Iterable[Tuple[_a, List[_b]]]) -> List[_a]:
    '''Topological sort, i.e. sort (non-uniquely) DAG nodes by directed path, e.g. sort packages by dependency order'''
    return foldq_(lambda acc, x: acc + [x[0]],
                  lambda acc, x, xs: [(a, [d for d in deps if d != x[0]]) for a, deps in xs],
                  lambda x: not x[1], nodes_incoming_edges_tuples, [])


def flatten(list_of_lists: Iterable[List]) -> List: return list(chain.from_iterable(list_of_lists))


def unzip(list_of_ntuples: Iterable[Iterable]) -> List[List]: return [list(t) for t in zip(*list_of_ntuples)]
def unzip_lazy(list_of_ntuples: Iterable[Iterable]) -> Iterator[List]: return map(list, zip(*list_of_ntuples))

def zip_maps(*maps: List[Mapping[_a, _b]]) -> Generator[Tuple[_a, Tuple], None, None]:
    for key in reduce(set.intersection, map(set, maps)): yield key, tuple(map(op.itemgetter(key), maps))


def unique(xs: Iterable[_a]) -> List[_a]:
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick
def unique_h(xs: Iterable[_a]) -> Set[_a]: return set(xs)


def eq_elems(xs: Iterable[_a], ys: Iterable[_a]) -> bool:
    cys = list(ys) # make a mutable copy
    try:
        for x in xs: cys.remove(x)
    except ValueError: return False
    return not cys
def eq_elems_h(xs: Iterable[_a], ys: Iterable[_a]) -> bool: return set(xs) == set(ys)


def diff(xs: Iterable[_a], ys: Iterable[_a]) -> List[_a]:
    cxs = list(xs) # make a mutable copy
    try:
        for y in ys: cxs.remove(y)
    except ValueError: pass
    return cxs
def diff_h(xs: Iterable[_a], ys: Iterable[_a]) -> Set[_a]: return set(xs) - set(ys)


def chunk(xs: Iterable[_a], n: int) -> Generator[_a, None, None]: return (xs[i:i + n] for i in range(0, len(xs), n))



## Specific-Purpose Function

# Simple Haskell convenience
def fst(ab: Tuple[_a, _b]) -> _a: return ab[0]
def snd(ab: Tuple[_a, _b]) -> _b: return ab[1]


def update_dict_with(d0: Dict, d1: Dict, f: Callable[[_a, _b], Union[_a, _b]]) -> Dict[Any, Union[_a, _b]]:
    '''Update a dictionary's entries with those of another using a given function, e.g. appending (operator.add is ideal for this)
    NOTE: This modifies d0, so might want to give it a deepcopy
    '''
    for k, v in d1.items(): d0[k] = f(d0[k], v) if k in d0 else v
    return d0


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


