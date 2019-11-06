from abc import ABC, abstractmethod
from Kernels.baseKernels import *
from collections import Counter


class KernelExpression(ABC):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def child_tree_context(self, child, T_left_F_right):
        '''
        Called by children
        :param child: the calling child (used by SumKE and ProductKE)
        :param T_left_F_right: string name of the field of this object in which the child resides (used by ChangeKE)
        :return: a closure building a copy of the full tree with an argument to go in the place of the calling child
        '''
        pass

    def to_kernel(self):
        return eval(str(self))

    @staticmethod
    def bs(str_expr):
        return '(' + str_expr + ')'


class SumKE(KernelExpression):
    def __init__(self, base_addenda, composite_addenda = [], parent: (KernelExpression, str) = None):
        super().__init__(parent)
        self.base_addenda = base_addenda if isinstance(base_addenda, Counter) else Counter(base_addenda)
        self.composite_addenda = composite_addenda
        self.simplify_base_addenda()

    def __str__(self):
        return ' + '.join([str(f) for f in list(self.base_addenda.elements()) + self.composite_addenda])

    def simplify(self):
        self.simplify_base_addenda()
        self.composite_addenda = [ca.simplify() for ca in self.composite_addenda]
        return self

    def child_tree_context(self, child, T_left_F_right): # T_left_F_right is used by ChangeKE
        other_composite_addenda = self.composite_addenda
        other_composite_addenda.remove(child)
        if self.parent is None:
            def pc(replacement_child):
                return SumKE(self.base_addenda, other_composite_addenda + [replacement_child], self.parent)
            return pc
        else:
            def pc(replacement_child):
                return self.parent[0].child_tree_context(self, self.parent[1])(
                    SumKE(self.base_addenda, other_composite_addenda + [replacement_child], self.parent)
                )
            return pc

    def simplify_base_addenda(self):
        # WN and C are addition-idempotent
        if self.base_addenda['WN'] > 1: self.base_addenda['WN'] = 1
        if self.base_addenda['C'] > 1: self.base_addenda['C'] = 1

    def new_base(self, new_base_addendum):
        self.base_addenda[new_base_addendum] += 1
        self.simplify_base_addenda()

# aTest = SumKE(['WN', 'WN', 'C', 'C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)


class ProductKE(KernelExpression):
    def __init__(self, base_factors, composite_factors = [], parent: (KernelExpression, str) = None):
        super().__init__(parent)
        self.base_factors = base_factors if isinstance(base_factors, Counter) else Counter(base_factors)
        self.composite_factors = composite_factors
        self.simplify_base_factors()

    def __str__(self):
        return ' * '.join([self.bs(str(f)) if isinstance(f, SumKE) else str(f) for f in list(self.base_factors.elements()) + self.composite_factors])

    def simplify(self):
        self.simplify_base_factors()
        self.composite_factors = [cf.simplify() for cf in self.composite_factors]
        return self

    def child_tree_context(self, child, T_left_F_right): # T_left_F_right is used by ChangeKE
        other_composite_factors = self.composite_factors
        other_composite_factors.remove(child)
        if self.parent is None:
            def pc(replacement_child):
                return ProductKE(self.base_factors, other_composite_factors + [replacement_child], self.parent)
            return pc
        else:
            def pc(replacement_child):
                return self.parent[0].child_tree_context(self, self.parent[1])(
                    ProductKE(self.base_factors, other_composite_factors + [replacement_child], self.parent)
                )
            return pc

    def simplify_base_factors(self):
        if self.base_factors['WN'] > 0: # WN acts as a multiplicative zero for all stationary kernels, i.e. all but LIN
            self.base_factors = Counter({'WN': 1, 'LIN': self.base_factors['LIN']})
        else:
            if self.base_factors['C'] > 0 and sum(self.base_factors.values()) - self.base_factors['C'] != 0: # If C is not the only present kernel
                self.base_factors['C'] = 0 # C is the multiplication-identity element
            elif self.base_factors['C'] > 1: self.base_factors['C'] = 1 # C is multiplication-idempotent
            if self.base_factors['SE'] > 1: self.base_factors['SE'] = 1 # SE is multiplication-idempotent

    def new_base(self, new_base_factor):
        self.base_factors[new_base_factor] += 1
        self.simplify_base_factors()

# aTest = ProductKE(['C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)
# aTest.new_base('WN')
# print(aTest)


class ChangeKE(KernelExpression):
    def __init__(self, CP_or_CW, left, right, parent: (KernelExpression, str) = None):
        super().__init__(parent)
        self.CP_or_CW = CP_or_CW
        self.left = left
        self.right = right

    def __str__(self):
        return self.CP_or_CW + self.bs(str(self.left) + ', ' + str(self.right))

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.left.right()
        return self

    def child_tree_context(self, child, T_left_F_right): # child used by SumKE and ProductKE
        if self.parent is None:
            if T_left_F_right:
                def pc(replacement_child):
                    return ChangeKE(self.CP_or_CW, replacement_child, self.right, self.parent)
                return pc
            else:
                def pc(replacement_child):
                    return ChangeKE(self.CP_or_CW, self.left, replacement_child, self.parent)
                return pc
        else:
            if T_left_F_right:
                def pc(replacement_child):
                    return self.parent[0].child_tree_context(self, self.parent[1])(
                        ChangeKE(self.CP_or_CW, replacement_child, self.right, self.parent)
                    )
                return pc
            else:
                def pc(replacement_child):
                    return self.parent[0].child_tree_context(self, self.parent[1])(
                        ChangeKE(self.CP_or_CW, self.left, replacement_child, self.parent)
                    )
                return pc




testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE(['SE', 'WN']))
testKern = testExpr.to_kernel()

print(testExpr)
print(testKern)

print(isinstance(testExpr, KernelExpression))
from GPy.kern.src.kern import Kern
print(isinstance(testKern, Kern))

