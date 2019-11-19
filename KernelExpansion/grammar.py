from KernelExpansion.kernelExpressionOperations import *
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

def standardise_singleton_root(k_expr_root): # Standardise a root if singleton: to SumKE of a base kernel or to the single composite term, making it the root
    simplified = k_expr_root.extract_if_singleton()
    if simplified is k_expr_root: return k_expr_root
    elif isinstance(simplified, str): return SumKE([simplified]).set_root()
    else: return standardise_singleton_root(simplified.set_parent(None).set_root())

def roots(k_expr_list):
    return [standardise_singleton_root(kex.root) for kex in k_expr_list]

def expand(k_expr, p_rules):
    return unique(roots(flatten([expand_node(kex, p_rules) for kex in k_expr.traverse()])))

def expand_node(k_expr, p_rules):
    return unique(flatten(flatten([pr(k_expr) for pr in p_rules])))

def deep_apply(operator, S, *args): # Deepcopy the tree and connect the new node to the rest of the tree (setting correct root and updating the parent if not root)
    return [S.new_tree_with_self_replaced(new_node) for new_node in operator(S, *args)]


## Production Rules

def plus_base(S): return [deep_apply(add, S, B) for B in base_kerns]
def times_base(S): return [deep_apply(multiply, S, B) for B in base_kerns - set('C')]
def replace_base(S): return [deep_apply(swap_base, S, B) for B in base_kerns]
def change_same(S): return [deep_apply(both_changes, S)]
def change_window_constant(S): return [deep_apply(one_change, S, 'CW', 'C')]
def times_shifted_base(S): return [deep_apply(multiply, S, SumKE([B, 'C'])) for B in base_kerns - set('C')]
def replace_with_singleton(S): return [deep_apply(replace_node, S, SumKE([B])) for B in base_kerns]
def remove_some_term(S): return [deep_apply(remove_a_term, S)]
production_rules_by_type = {
    'basic': {
        'plus_base': plus_base, # S -> S + B
        'times_base': times_base, # S -> S * B
        'replace_base': replace_base, # B -> B
    }, 'change': {
        'change_same': change_same, # S -> CP(S, S) and S -> CW(S, S)
        'change_window_constant': change_window_constant, # S -> CW(S, C) and S -> CW(C, S)
    }, 'heuristic': {
        'times_shifted_base': times_shifted_base, # S -> S * (B + C)
        'replace_with_singleton': replace_with_singleton, # S -> B
        'remove_some_term': remove_some_term, # S + S2 -> S and S * S2 -> S
    }
}
production_rules_all = flatten([list(x.values()) for x in production_rules_by_type.values()])


# TODO:
#   Decide whether all rules should be used at each depth or whether some types should be dropped or introduced only at some point
