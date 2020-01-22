from itertools import chain
from functools import reduce


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def unique(xs): # Using a list instead of a set because of requirement to work with unhashables
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick


def partition(p, xs): # Haskell's partition function: partition p xs == (filter p xs, filter (not . p) xs)
    def select(acc, x):
        acc[not p(x)].append(x)
        return acc
    return reduce(select, xs, ([],[]))


def lists_of_unhashables__eq(xs, ys):
    cys = list(ys) # make a mutable copy
    try:
        for x in xs: cys.remove(x)
    except ValueError: return False
    return not cys

def lists_of_unhashables__diff(xs, ys):
    cxs = list(xs) # make a mutable copy
    try:
        for y in ys: cxs.remove(y)
    except ValueError: pass
    return cxs

def diff(xs, ys): return list(set(xs) - set(ys))


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2: return {t1: [k1, k2]}
    else: return {t1: k1, t2: k2}


def interval_overlap(a, b): # Two interval tuples
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


# Update a dictionary's entries with those of another using a given function, e.g. appending (operator.add is ideal for this)
# NOTE: This modifies d0, so might want to give it a deepcopy
def update_dict_with(d0, d1, f):
    for k, v in d1.items(): d0[k] = f(d0[k], v) if k in d0 else v
    return d0
