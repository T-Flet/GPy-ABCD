from GPy_ABCD.KernelExpansion.kernelExpression import *
from GPy_ABCD.Util.genericUtil import sortOutTypePair


# NICE BUT UNNECESSARY REFACTORING: make add and multiply two cases of the same function

# NOTE:
#   copy and deepcopy from copy would look very elegant when partenered with methods that res = self in some cases below, BUT:
#       not using copy because it would affect the original ks non-base-type field values
#       not using deepcopy because it would create much more than the required fields

# POLICY
#   Related to the above note: deepcopies are left to KernelExpression methods (including __init__) and, if required,
#   to larger routines (which might use these operators)
#   ALSO, IMPORTANTLY AND CONSEQUENTLY: no operator is to modify either argument in any way


def add(k1, k2, with_nested_case = True): # Simple addition, NOT DISTRIBUTING OR SIMPLIFYING
    # with_nested_case = True returns a list instead of a single result; only str + ProductKE will be a non-singleton
    res = None
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            res = SumKE(+k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms)
        elif isinstance(k1, ProductKE):
            res = SumKE([], [k1, k2])
        elif isinstance(k1, ChangeKE):
            res = SumKE([], [k1, k2])
        else: # elif isinstance(k1, str):
            res = SumKE([k1, k2], [])
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                res = SumKE(+pair[SumKE].base_terms + Counter([pair[str]]), pair[SumKE].composite_terms)
            elif ProductKE in pair.keys():
                res = SumKE([pair[str]], [pair[ProductKE]])
                if with_nested_case: # Also add the base kernel to each base_term factor separately
                    return [res] + [ProductKE(+pair[ProductKE].base_terms - Counter([bt]), pair[ProductKE].composite_terms + [SumKE([pair[str], bt], [])]) for bt in pair[ProductKE].base_terms.elements()]
            else: # elif ChangeKE in pair.keys():
                res = SumKE([pair[str]], [pair[ChangeKE]])
        elif pair.keys() == {SumKE, ProductKE}:
            res = SumKE(+pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ProductKE]])
        elif pair.keys() == {SumKE, ChangeKE}:
            res = SumKE(+pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ChangeKE]])
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            res = SumKE([], list(pair.values()))
    return [res] if with_nested_case else res


def multiply(k1, k2, with_nested_case = True): # Simple multiplication, NOT DISTRIBUTING OR SIMPLIFYING
    # with_nested_case = True returns a list instead of a single result; only str + SumKE will be a non-singleton
    res = None
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            res = ProductKE([], [k1, k2])
        elif isinstance(k1, ProductKE):
            res = ProductKE(+k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms)
        elif isinstance(k1, ChangeKE):
            res = ProductKE([], [k1, k2])
        else: # elif isinstance(k1, str):
            res = ProductKE([k1, k2], [])
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                res = ProductKE([pair[str]], [pair[SumKE]])
                if with_nested_case:  # Also multiply the base kernel with each base_term addendum separately
                    return [res] + [SumKE(+pair[SumKE].base_terms - Counter([bt]), pair[SumKE].composite_terms + [ProductKE([pair[str], bt], [])]) for bt in pair[SumKE].base_terms.elements()]
            elif ProductKE in pair.keys():
                res = ProductKE(+pair[ProductKE].base_terms + Counter([pair[str]]), pair[ProductKE].composite_terms)
            else: # elif ChangeKE in pair.keys():
                res = ProductKE([pair[str]], [pair[ChangeKE]])
        elif pair.keys() == {SumKE, ProductKE}:
            res = ProductKE(+pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[SumKE]])
        elif pair.keys() == {SumKE, ChangeKE}:
            res = ProductKE([], list(pair.values()))
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            res = ProductKE(+pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[ChangeKE]])
    return [res] if with_nested_case else res


def swap_base(S, B):
    res = []
    if isinstance(S, SumKE) or isinstance(S, ProductKE):
        res = [type(S)((+S.base_terms) + Counter({bt: -1, B: 1}), S.composite_terms) for bt in (+S.base_terms).keys()]
    elif isinstance(S, ChangeKE):
        res += [ChangeKE(S.CP_or_CW, B, S.right)] if isinstance(S.left, str) and S.left != B else []
        res += [ChangeKE(S.CP_or_CW, S.left, B)] if isinstance(S.right, str) and S.right != B else []
    else: # elif isinstance(k1, str): # This never occurs in the expansion though
        res = [B]
    return res


def one_change(S, CP_or_CW, S2 = None):
    if S2 is None or S2 == S: return [ChangeKE(CP_or_CW, S, S)]
    else: return [ChangeKE(CP_or_CW, S, S2), ChangeKE(CP_or_CW, S2, S)]


def both_changes(S, S2 = None):
    res = []
    for CP_or_CW in ('CP', 'CW'):
        if S2 is None or S2 == S: res.append(ChangeKE(CP_or_CW, S, S))
        else: res += [ChangeKE(CP_or_CW, S, S2), ChangeKE(CP_or_CW, S2, S)]
    return res


def replace_node(S, S2):
    return [SumKE([S2])] if isinstance(S2, str) else [S2]


def remove_a_term(S):
    res = []
    if isinstance(S, SumKE) or isinstance(S, ProductKE):
        if S.term_count() > 1:
            res += [type(S)((+S.base_terms) + Counter({bt: -1}), S.composite_terms) for bt in (+S.base_terms).keys()]
            res += [type(S)(+S.base_terms, S.composite_terms[0:cti] + S.composite_terms[(cti+1):]) for cti in range(len(S.composite_terms))]
        # elif not S.is_root(): print('THIS WAS HERE JUST TO VERIFY THAT IT NEVER HAPPENS (BY DESIGN); NOT THAT IT WOULD BREAK ANYTHING IF IT DID')
    elif isinstance(S, ChangeKE):
        res = [SumKE([branch]) if isinstance(branch, str) else branch for branch in (S.left, S.right)]
    else: # elif isinstance(k1, str): # This never occurs in the expansion though
        res = [[]]
    return res
