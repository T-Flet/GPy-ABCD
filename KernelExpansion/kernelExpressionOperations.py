from KernelExpansion.kernelExpression import *


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2:
        return {t1: [k1, k2]}
    else:
        return {t1: k1, t2: k2}


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
            res = SumKE(k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms)
        elif isinstance(k1, ProductKE):
            res = SumKE([], [k1, k2])
        elif isinstance(k1, ChangeKE):
            res = SumKE([], [k1, k2])
        else: # elif isinstance(k1, str):
            res = SumKE([k1, k2], [])
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                res = SumKE(pair[SumKE].base_terms + Counter([pair[str]]), pair[SumKE].composite_terms)
            elif ProductKE in pair.keys():
                res = SumKE([pair[str]], [pair[ProductKE]])
                if with_nested_case: # Also add the base kernel to each base_term factor separately
                    return [res] + [ProductKE(pair[ProductKE].base_terms - Counter([bt]), pair[ProductKE].composite_terms + [SumKE([pair[str], bt], [])]) for bt in pair[ProductKE].base_terms.elements()]
            else: # elif ChangeKE in pair.keys():
                res = SumKE([pair[str]], [pair[ChangeKE]])
        elif pair.keys() == {SumKE, ProductKE}:
            res = SumKE(pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ProductKE]])
        elif pair.keys() == {SumKE, ChangeKE}:
            res = SumKE(pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ChangeKE]])
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
            res = ProductKE(k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms)
        elif isinstance(k1, ChangeKE):
            res = ProductKE([], [k1, k2])
        else: # elif isinstance(k1, str):
            res = ProductKE([k1, k2], [])
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                res = ProductKE([pair[str]], [pair[SumKE]])
                if with_nested_case:  # Also multiply the base kernel with each base_term addendum separately
                    return [res] + [SumKE(pair[SumKE].base_terms - Counter([bt]), pair[SumKE].composite_terms + [ProductKE([pair[str], bt], [])]) for bt in pair[SumKE].base_terms.elements()]
            elif ProductKE in pair.keys():
                res = ProductKE(pair[ProductKE].base_terms + Counter([pair[str]]), pair[ProductKE].composite_terms)
            else: # elif ChangeKE in pair.keys():
                res = ProductKE([pair[str]], [pair[ChangeKE]])
        elif pair.keys() == {SumKE, ProductKE}:
            res = ProductKE(pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[SumKE]])
        elif pair.keys() == {SumKE, ChangeKE}:
            res = ProductKE([], list(pair.values()))
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            res = ProductKE(pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[ChangeKE]])
    return [res] if with_nested_case else res