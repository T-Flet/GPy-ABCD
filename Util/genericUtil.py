from itertools import chain


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def unique(xs): # Using a list instead of a set because of requirement to work with unhashables
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2:
        return {t1: [k1, k2]}
    else:
        return {t1: k1, t2: k2}


def interval_overlap(a, b): # Two interval tuples
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

