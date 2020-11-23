from GPy_ABCD.config import __INCLUDE_SE_KERNEL, __USE_LIN_KERNEL_HORIZONTAL_OFFSET
from GPy_ABCD.KernelExpressions.all import KernelExpression
from GPy_ABCD.KernelExpansion.kernelExpressionOperations import *
from GPy_ABCD.Kernels.baseKernels import base_kerns
from GPy_ABCD.Util.genericUtil import flatten, unique


## Expansion functions

def standardise_singleton_root(k_expr_root): # Standardise a root if singleton: to SumKE of a base kernel or to the single composite term, making it the root
    simplified = k_expr_root.simplify().extract_if_singleton()
    if simplified is k_expr_root: return k_expr_root
    elif isinstance(simplified, str): return SumKE([simplified]).set_root()
    else: return standardise_singleton_root(simplified.set_parent(None).set_root())

def roots(k_expr_list): return [standardise_singleton_root(kex.root) for kex in k_expr_list]

def expand(k_expr, p_rules): return unique(roots(flatten([expand_node(kex, p_rules) for kex in k_expr.traverse()])))

def expand_node(k_expr, p_rules): return unique(flatten(flatten([pr(k_expr) for pr in p_rules])))

# Deepcopy the tree and connect the new node to the rest of the tree (setting correct root and updating the parent if not root)
def deep_apply(operator, S, *args): return [S.new_tree_with_self_replaced(new_node) for new_node in operator(S, *args)]



## Production Rules ##

base_kerns_for_prod = base_kerns if __INCLUDE_SE_KERNEL else base_kerns - {'SE'}

# Basic
def plus_base(S):
    '''S -> S + B'''
    return [deep_apply(add, S, B) for B in base_kerns_for_prod]
def times_base(S):
    '''S -> S * B'''
    return [deep_apply(multiply, S, B) for B in base_kerns_for_prod - {'C'}]
def replace_base(S):
    '''B -> B'''
    return [deep_apply(swap_base, S, B) for B in base_kerns_for_prod]

# Change
def change_new_base(S):
    '''S -> CP(S, B) and S -> CW(S, B)'''
    return [deep_apply(both_changes, S, B) for B in base_kerns_for_prod - {'C'}]
def change_same(S):
    '''S -> CP(S, S) and S -> CW(S, S)'''
    return [deep_apply(both_changes, S)]
def change_window_constant(S):
    '''S -> CW(S, C) and S -> CW(C, S)'''
    return [deep_apply(one_change, S, 'CW', 'C')]
def change_window_linear(S):
    '''S -> CW(S, LIN) and S -> CW(LIN, S)'''
    return [deep_apply(one_change, S, 'CW', 'LIN')]
def change_point_linear(S):
    '''S -> CP(S, LIN) and S -> CP(LIN, S)'''
    return [deep_apply(one_change, S, 'CP', 'LIN')]

# Heuristic
def times_shifted_base(S):
    '''S -> S * (B + C)'''
    return [deep_apply(multiply, S, SumKE([B, 'C'])) for B in base_kerns_for_prod - {'C'}]
def replace_with_singleton(S):
    '''S -> B'''
    return [deep_apply(replace_node, S, SumKE([B])) for B in base_kerns_for_prod]
def remove_some_term(S):
    '''S + S2 -> S and S * S2 -> S'''
    return [deep_apply(remove_a_term, S)]
def try_higher_curves(S):
    '''S -> S with 2nd or 3rd order polynomials or PER * PER, discouraging SE'''
    return [deep_apply(higher_curves, S)]

# TODO:
#   - Have a way to refer to production rules which depend on the user-definable base_kerns_for_prod

production_rules_by_type = {
    'basic': {
        'plus_base': plus_base,
        'times_base': times_base,
        'replace_base': replace_base,
    }, 'change': {
        'change_new_base': change_new_base,
        'change_same': change_same,
        'change_point_linear': change_point_linear,
        'change_window_constant': change_window_constant,
        'change_window_linear': change_window_linear,
    }, 'heuristic': {
        'times_shifted_base': times_shifted_base,
        'replace_with_singleton': replace_with_singleton,
        'remove_some_term': remove_some_term,
        'try_higher_curves': try_higher_curves
    }
}
non_grouped_prs = {k: v for d in production_rules_by_type.values() for k, v in d.items()}


# Default groups

production_rules = {
    'All': list(non_grouped_prs.values()),
    'Original_ABCD': [plus_base, times_base, replace_base, change_same, change_window_constant, times_shifted_base, replace_with_singleton, remove_some_term],
    'Minimal': [plus_base, times_base, replace_base, change_same],
    'Default': [plus_base, times_base, replace_base, change_point_linear, change_window_linear, times_shifted_base, replace_with_singleton, remove_some_term]
}



## Start Kernels ##

def pseudo_to_real_kex(pkex):
    if isinstance(pkex, KernelExpression): return pkex
    elif pkex in base_kerns: return SumKE([pkex])
    else: raise ValueError(f'Given pseudo-kernel-expression value could not be parsed: {pkex}')
def make_simple_kexs(pseudo_kexs): return [pseudo_to_real_kex(pkex)._initialise() for pkex in pseudo_kexs]


# Default groups

start_kernels = {
    'Default': make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
                                          [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
                                          both_changes('LIN')), # To catch a possible changepoint or changewindow with simple enough shapes
    'Extended': make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
                                          [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
                                          [SumKE(['C'], [ck]) for ck in both_changes('LIN')]), # To catch a possible changepoint or changewindow with simple enough shapes
    'Original_ABCD': make_simple_kexs(['WN'])
}


if not __USE_LIN_KERNEL_HORIZONTAL_OFFSET: # Non-offset-LIN needs C to achieve (almost) parity with offset-LIN
    for sks in start_kernels.values(): sks += make_simple_kexs([SumKE(['LIN', 'C'])])


