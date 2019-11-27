from itertools import chain

def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))

def unique(xs): # Using a list instead of a set because of requirement to work with unhashables
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick