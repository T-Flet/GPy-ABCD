from Kernels.baseKernels import *
from kernelExpression import *
from kernelExpressionOperations import *
from copy import deepcopy
from itertools import chain


base_kerns = frozenset(['WN', 'C', 'LIN', 'SE', 'PER'])
# stationary_kerns = frozenset(['WN', 'C', 'SE', 'PER'])
# addition_idempotent_kerns = frozenset(['WN', 'C'])
# multiplication_idempotent_kerns = frozenset(['WN', 'C', 'SE'])
# multiplication_zero_kerns = frozenset(['WN']) # UNLESS LIN!!!!!!! I.E. ZERO ONLY FOR STATIONARY KERNELS
# multiplication_identity_kerns = frozenset(['C'])

base_order = {'PER': 1, 'WN': 2, 'SE': 3, 'C': 4, 'LIN': 5}
    # Then sort by: sorted(LIST, key=lambda SYM: baseOrder[SYM])


## Utility functions

def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))

def unique(xs):
    seen = [] # Note: 'in' tests x is z or x == z, hence it works with __eq__ overloading
    return [x for x in xs if x not in seen and not seen.append(x)] # Neat short-circuit 'and' trick


## Expansion functions

def roots(k_expr_list):
    return [kex.root for kex in k_expr_list]

def expand(k_expr, p_rules):
    return unique(roots(flatten([expand_node(kex, p_rules) for kex in k_expr.traverse()])))

def expand_node(k_expr, p_rules):
    return unique(flatten(flatten([pr(k_expr) for pr in p_rules])))

## Add more simplifications, e.g. zeros or identities with composites (singleton WN * composite)
## Add already-tested-expression caching and filter here

def deep_apply(operator, S, *args): # Deepcopy the tree and connect the new node to the rest of the tree (setting correct root and updating the parent if not root)
    return [S.new_tree_with_self_replaced(new_node) for new_node in operator(S, *args)]


## Production Rules

def plus_base(S): return [deep_apply(add, S, B) for B in base_kerns]
def times_base(S): return [deep_apply(multiply, S, B) for B in base_kerns - set('C')]
# def replace_base(S): return [swap_base(S, B) for B in base_kerns]
production_rules = { # IS A LIST BETTER BECAUSE KEYS ARE NEVER USED?
    'plus_base': plus_base,
    'times_base': times_base,
    # 'replace_base': replace_base
}




# LHS:
#   S: B | CP()  | CW() | ( .. )
#   B: WN, C, LIN, SE, PER
prod_rules_to_implement = [
    ('S', 'S + B'),
    ('S', 'S * B'),
    ('B', 'B'),
    ('S', 'CP(S, S)'),
    ('S', 'CP(S, C)'),
    ('S', 'CP(C, S)'),
    ('S', 'CW(S, S)'),
    ('S', 'CW(S, C)'),
    ('S', 'CW(C, S)'),
    ('S', 'S * (B + C)'),
    ('S', 'B'),
    ('S + S', 'S'),
    ('S * S', 'S'),
    ('S', '')
]




