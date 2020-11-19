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

def roots(k_expr_list):
    return [standardise_singleton_root(kex.root) for kex in k_expr_list]

def expand(k_expr, p_rules):
    return unique(roots(flatten([expand_node(kex, p_rules) for kex in k_expr.traverse()])))

def expand_node(k_expr, p_rules):
    return unique(flatten(flatten([pr(k_expr) for pr in p_rules])))

def deep_apply(operator, S, *args): # Deepcopy the tree and connect the new node to the rest of the tree (setting correct root and updating the parent if not root)
    return [S.new_tree_with_self_replaced(new_node) for new_node in operator(S, *args)]


## Production Rules

base_kerns_for_prod = base_kerns if __INCLUDE_SE_KERNEL else base_kerns - {'SE'}

def plus_base(S): return [deep_apply(add, S, B) for B in base_kerns_for_prod]
def times_base(S): return [deep_apply(multiply, S, B) for B in base_kerns_for_prod - {'C'}]
def replace_base(S): return [deep_apply(swap_base, S, B) for B in base_kerns_for_prod]
def change_new_base(S): return [deep_apply(both_changes, S, B) for B in base_kerns_for_prod - {'C'}] # Not in original ABCD
def change_same(S): return [deep_apply(both_changes, S)]
def change_window_constant(S): return [deep_apply(one_change, S, 'CW', 'C')]
def change_window_linear(S): return [deep_apply(one_change, S, 'CW', 'LIN')] # Not in original ABCD
def change_point_linear(S): return [deep_apply(one_change, S, 'CP', 'LIN')] # Not in original ABCD
def times_shifted_base(S): return [deep_apply(multiply, S, SumKE([B, 'C'])) for B in base_kerns_for_prod - {'C'}]
def replace_with_singleton(S): return [deep_apply(replace_node, S, SumKE([B])) for B in base_kerns_for_prod]
def remove_some_term(S): return [deep_apply(remove_a_term, S)]
def try_higher_curves(S): return [deep_apply(higher_curves, S)] # Not in original ABCD


# TODO:
#   - Group production rules by introduced base kernels in order to be able to remove some, in particular changepoint ones
#   - Have a way to refer to the ones depending on the user-definable base_kerns_for_prod

production_rules_by_type = {
    'basic': {
        'plus_base': plus_base, # S -> S + B
        'times_base': times_base, # S -> S * B
        'replace_base': replace_base, # B -> B
    }, 'change': {
        # 'change_new_base': change_new_base, # S -> CP(S, B) and S -> CW(S, B)
        # 'change_same': change_same, # S -> CP(S, S) and S -> CW(S, S)
        'change_point_linear': change_point_linear, # S -> CP(S, LIN) and S -> CP(LIN, S)
        # 'change_window_constant': change_window_constant, # S -> CW(S, C) and S -> CW(C, S)
        'change_window_linear': change_window_linear, # S -> CW(S, LIN) and S -> CW(LIN, S)
    }, 'heuristic': {
        'times_shifted_base': times_shifted_base, # S -> S * (B + C)
        'replace_with_singleton': replace_with_singleton, # S -> B
        'remove_some_term': remove_some_term, # S + S2 -> S and S * S2 -> S
        # 'try_higher_curves': try_higher_curves, # S -> S with higher polynomials or PERs, discouraging SE
    }
}



## Reasonable Start Kernels & Production Rules

# Kernels

def pseudo_to_real_kex(pkex):
    if isinstance(pkex, KernelExpression): return pkex
    elif pkex in base_kerns: return SumKE([pkex])
    else: raise ValueError(f'Given pseudo-kernel-expression value could not be parsed: {pkex}')
def make_simple_kexs(pseudo_kexs): return [pseudo_to_real_kex(pkex)._initialise() for pkex in pseudo_kexs]


standard_start_kernels = make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
                                          [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
                                          both_changes('LIN')) # To catch a possible changepoint or changewindow with simple enough shapes

extended_start_kernels = make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
                                          [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
                                          [SumKE(['C'], [ck]) for ck in both_changes('LIN')]) # To catch a possible changepoint or changewindow with simple enough shapes

test_start_kernels = make_simple_kexs(list(base_kerns - {'SE'}) + # Base Kernels without SE
                                          [ProductKE(['LIN', 'LIN']), ProductKE(['LIN', 'LIN', 'LIN']), SumKE(['PER', 'C'])] + # More generic LIN and PER
                                          both_changes('LIN')) # To catch a possible changepoint or changewindow with simple enough shapes

if not __USE_LIN_KERNEL_HORIZONTAL_OFFSET: assert SumKE(['LIN', 'C']) in standard_start_kernels, 'Non-offset-LIN needs C to achieve (almost) parity with offset-LIN'


# Production Rules

production_rules_all = flatten([list(x.values()) for x in production_rules_by_type.values()])
# production_rules_start = [plus_base, times_base, replace_base, change_new_base]
# production_rules_start = list(production_rules_by_type['basic'].values()) + [change_same]


