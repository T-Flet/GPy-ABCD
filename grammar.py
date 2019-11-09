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




def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))

def expand_node(k_expr, p_rules):
    return flatten([pr(k_expr) for pr in p_rules])

def expand(k_expr, p_rules):
    return flatten([expand_node(kex, p_rules) for kex in k_expr.traverse()])


## Add more simplifications
## Add duplicate removal and already-tested-expression filter here; need to implmenet equality comparison (not overloading __eq__ though)


def deep_apply(operator, S, *args): # Deepcopy the tree and connect the new node to the rest of the tree
    copied_node = deepcopy(S)
    new_node = operator(copied_node, *args).set_parent(copied_node.parent)
    if copied_node.root == copied_node:
        new_node.set_root(new_node)
    else:
        new_node.set_root(copied_node.root)
        new_node.parent.reassign_child(copied_node, new_node)
    return new_node


def plus_base(S): return [deep_apply(add, S, B) for B in base_kerns]
def times_base(S): return [deep_apply(multiply, S, B) for B in base_kerns]
# def replace_base(S): return [swap_base(S, B) for B in base_kerns]
production_rules = { # IS A LIST BETTER BECAUSE KEYS ARE NEVER USED?
    'plus_base': plus_base,
    'times_base': times_base,
    # 'replace_base': replace_base
}


# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))._initialise()
# a = expand_node(testExpr, production_rules.values())
# a = expand(testExpr, production_rules.values())
# for e in a:
#     print(e)





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




