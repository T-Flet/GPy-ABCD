from kernelExpression import *
from copy import copy


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2:
        return {t1: [k1, k2]}
    else:
        return {t1: k1, t2: k2}


def add(k1, k2, root = None): # Simple addition, NOT DISTRIBUTING OR SIMPLIFYING
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            return copy(k1).new_base(k2.base_terms).new_composite(k2.composite_terms).set_root(root)
        elif isinstance(k1, ProductKE):
            return SumKE([], [k1, k2], root)
        elif isinstance(k1, ChangeKE):
            return SumKE([], [k1, k2], root)
        else: # elif isinstance(k1, str):
            return SumKE([k1, k2], [], root)
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                return copy(pair[SumKE]).new_base(pair[str]).set_root(root)
            elif ProductKE in pair.keys():
                return SumKE([pair[str]], [pair[ProductKE]], root)
            else: # elif ChangeKE in pair.keys():
                return SumKE([pair[str]], [pair[ChangeKE]], root)
        elif pair.keys() == {SumKE, ProductKE}:
            return copy(pair[SumKE]).new_composite(pair[ProductKE]).set_root(root)
        elif pair.keys() == {SumKE, ChangeKE}:
            return copy(pair[SumKE]).new_composite(pair[ChangeKE]).set_root(root)
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            return SumKE([], list(pair.values()), root)


def multiply(k1, k2, root = None): # Simple multiplication, NOT DISTRIBUTING OR SIMPLIFYING
    pair = sortOutTypePair(k1, k2)
    if len(pair) == 1:
        if isinstance(k1, SumKE):
            return ProductKE([], [k1, k2], root)
        elif isinstance(k1, ProductKE):
            return copy(k1).new_base(k2.base_terms).new_composite(k2.composite_terms).set_root(root)
        elif isinstance(k1, ChangeKE):
            return ProductKE([], [k1, k2], root)
        else: # elif isinstance(k1, str):
            return ProductKE([k1, k2], [], root)
    else:
        if str in pair.keys():
            if SumKE in pair.keys():
                return ProductKE([pair[str]], [pair[SumKE]], root)
            elif ProductKE in pair.keys():
                return copy(pair[ProductKE]).new_base(pair[str]).set_root(root)
            else: # elif ChangeKE in pair.keys():
                return ProductKE([pair[str]], [pair[ChangeKE]], root)
        elif pair.keys() == {SumKE, ProductKE}:
            return copy(pair[ProductKE]).new_composite(pair[SumKE]).set_root(root)
        elif pair.keys() == {SumKE, ChangeKE}:
            return ProductKE([], list(pair.values()), root)
        else: # elif pair.keys() == {ProductKE, ChangeKE}:
            return copy(pair[ProductKE]).new_composite(pair[ChangeKE]).set_root(root)
