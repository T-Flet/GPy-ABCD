from kernelExpression import *


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2:
        return {t1: [k1, k2]}
    else:
        return {t1: k1, t2: k2}


# NOTE:
#   copy and deepcopy from copy would look very elegant when partenered with methods that return self in some cases below, BUT:
#       not using copy because it would affect the original ks non-base-type field values
#       not using deepcopy because it would create much more than the required fields


def add(k1, k2, root = None): # Simple addition, NOT DISTRIBUTING OR SIMPLIFYING
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            return SumKE(k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms, root)
        elif isinstance(k1, ProductKE):
            return SumKE([], [k1, k2], root)
        elif isinstance(k1, ChangeKE):
            return SumKE([], [k1, k2], root)
        else: # elif isinstance(k1, str):
            return SumKE([k1, k2], [], root)
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                return SumKE(pair[SumKE].base_terms + Counter([pair[str]]), pair[SumKE].composite_terms, root)
            elif ProductKE in pair.keys():
                return SumKE([pair[str]], [pair[ProductKE]], root)
            else: # elif ChangeKE in pair.keys():
                return SumKE([pair[str]], [pair[ChangeKE]], root)
        elif pair.keys() == {SumKE, ProductKE}:
            return SumKE(pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ProductKE]], root)
        elif pair.keys() == {SumKE, ChangeKE}:
            return SumKE(pair[SumKE].base_terms, pair[SumKE].composite_terms + [pair[ChangeKE]], root)
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            return SumKE([], list(pair.values()), root)


def multiply(k1, k2, root = None): # Simple multiplication, NOT DISTRIBUTING OR SIMPLIFYING
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            return ProductKE([], [k1, k2], root)
        elif isinstance(k1, ProductKE):
            return ProductKE(k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms, root)
        elif isinstance(k1, ChangeKE):
            return ProductKE([], [k1, k2], root)
        else: # elif isinstance(k1, str):
            return ProductKE([k1, k2], [], root)
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                return ProductKE([pair[str]], [pair[SumKE]], root)
            elif ProductKE in pair.keys():
                return ProductKE(pair[ProductKE].base_terms + Counter([pair[str]]), pair[ProductKE].composite_terms, root)
            else: # elif ChangeKE in pair.keys():
                return ProductKE([pair[str]], [pair[ChangeKE]], root)
        elif pair.keys() == {SumKE, ProductKE}:
            return ProductKE(pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[SumKE]], root)
        elif pair.keys() == {SumKE, ChangeKE}:
            return ProductKE([], list(pair.values()), root)
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            return ProductKE(pair[ProductKE].base_terms, pair[ProductKE].composite_terms + [pair[ChangeKE]], root)
