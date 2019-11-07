from abc import ABC, abstractmethod
from Kernels.baseKernels import *
from collections import Counter
from itertools import chain
# import kernelExpressionOperations


# IDEA:
#   Models are represented by a tree of KernelExpressions, which can be Sum, Product and Change nodes, each of which may contain raw strings as leafs.
#   The tree root is stored in each node, then from any traversal one can just deepcopy single nodes to get attached whole trees for free.
#   When expanding a tree: can apply all production rules at internal nodes, but only base-kernel ones at the leaves;
#   the mult/sum of sum/mult rules are applied to leaves FROM their parents; this covers the whole tree



class KernelExpression(ABC): # Abstract
    def __init__(self, root):
        super().__init__()
        self.root = root

    ### Overloading with functions that require determingin which subclass of this one each operand is does not seem possible; generic wrapper class?
    # def __add__(self, kex):
    #     return kernelExpressionOperations.add(self, kex)
    # 
    # def __mul__(self, kex):
    #     return kernelExpressionOperations.multiply(self, kex)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def simplify(self):
        pass

    # @abstractmethod
    # def sum_of_prods_form(self):
    #     pass

    @abstractmethod
    def traverse(self):
        pass

    @abstractmethod
    def set_root(self, new_root):
        pass

    def to_kernel(self):
        return eval(str(self))

    @staticmethod
    def bs(str_expr):
        return '(' + str_expr + ')'


class SumOrProductKE(KernelExpression): # Abstract
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, symbol = '+'):
        super().__init__(root)
        self.base_terms = base_terms if isinstance(base_terms, Counter) else Counter(base_terms)
        self.composite_terms = composite_terms
        self.symbol = symbol
        self.simplify_base_terms()

    def __str__(self):
        return (' ' + self.symbol + ' ').join([str(f) for f in list(self.base_terms.elements()) + self.composite_terms])

    def simplify(self):
        self.simplify_base_terms()
        self.composite_terms = [ca.simplify() for ca in self.composite_terms]
        return self

    @abstractmethod
    def simplify_base_terms(self):
        pass

    def traverse(self):
        return [self] + list(chain.from_iterable([ca.traverse() for ca in self.composite_terms]))

    def set_root(self, new_root):
        self.root = new_root
        for cf in self.composite_terms: cf.set_root(new_root)
        return self

    def new_base(self, new_base_terms):
        if isinstance(new_base_terms, str):
            self.base_terms[new_base_terms] += 1
        else: # list or Counter
            self.base_terms.update(new_base_terms)
        self.simplify_base_terms()
        return self

    def new_composite(self, new_composite_terms):
        if isinstance(new_composite_terms, KernelExpression):
            self.composite_terms += [new_composite_terms]
        else: # list
            self.composite_terms += new_composite_terms
        return self


class SumKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, '+')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1


class ProductKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, '*')

    def simplify_base_terms(self):
        if self.base_terms['WN'] > 0: # WN acts as a multiplicative zero for all stationary kernels, i.e. all but LIN
            self.base_terms = Counter({'WN': 1, 'LIN': self.base_terms['LIN']})
        else:
            if self.base_terms['C'] > 0 and sum(self.base_terms.values()) - self.base_terms['C'] != 0: # If C is not the only present kernel
                self.base_terms['C'] = 0 # C is the multiplication-identity element
            elif self.base_terms['C'] > 1: self.base_terms['C'] = 1 # C is multiplication-idempotent
            if self.base_terms['SE'] > 1: self.base_terms['SE'] = 1 # SE is multiplication-idempotent


class ChangeKE(KernelExpression):
    def __init__(self, CP_or_CW, left, right, root: KernelExpression = None):#parent: (KernelExpression, str) = None):
        super().__init__(root)
        self.CP_or_CW = CP_or_CW
        self.left = left
        self.right = right

    def __str__(self):
        return self.CP_or_CW + self.bs(str(self.left) + ', ' + str(self.right))

    def simplify(self):
        if isinstance(self.left, KernelExpression): self.left.simplify()
        if isinstance(self.right, KernelExpression): self.right.simplify()
        return self

    def traverse(self):
        res = [self]
        res += self.left.traverse() if isinstance(self.left, KernelExpression) else [self.left]
        res += self.right.traverse() if isinstance(self.right, KernelExpression) else [self.right]
        return res

    def set_root(self, new_root):
        self.root = new_root
        if isinstance(self.left, KernelExpression): self.left.set_root(new_root)
        if isinstance(self.right, KernelExpression): self.right.set_root(new_root)
        return self
