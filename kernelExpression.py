from abc import ABC, abstractmethod
from Kernels.baseKernels import *
from collections import Counter
from itertools import chain
from functools import reduce
# import kernelExpressionOperations


# IDEA:
#   Models are represented by a tree of KernelExpressions, which can be Sum, Product and Change nodes;
#   each type of node may contain raw strings base kernels or composite ones, however bare raw string leaves may only occur in ChangeKEs.
#   The tree root is stored in each node, as is the direct parent, therefore from any traversal one can just deepcopy
#   single nodes to get the rest of the tree for free.
#   When expanding a tree: can apply all production rules at internal nodes, but only base-kernel ones at the leaves;
#   the mult/sum of sum/mult rules are applied to leaves FROM their parents; this covers the whole tree
#   Both traverse and reduce ignore bare string leaves and assume the user handles them from their ChangeKE parent.



class KernelExpression(ABC): # Abstract
    def __init__(self, root, parent):
        super().__init__()
        self.root = root
        self.parent = parent

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

    # NOTE: both traverse and reduce ignore raw-string leaves (which can only happen in ChangeKEs);
    #       care has to be taken to perform required operations on them from their parent
    @abstractmethod
    def traverse(self):
        pass

    # Same note as traverse; see Test.checkKernelExpressions for an example func
    @abstractmethod
    def reduce(self, func, acc):
        pass

    @abstractmethod
    def set_root(self, new_root):
        pass

    def set_parent(self, new_parent):
        self.parent = new_parent
        return self

    @abstractmethod
    def _set_all_parents(self):
        pass

    @abstractmethod
    def _check_all_parents(self):
        pass

    # NOTE: The argument is intended to be the root of a manually coded tree
    def _initialise(self):
        return self.set_root(self)._set_all_parents()

    @abstractmethod
    def reassign_child(self, old_child, new_child):
        pass

    def to_kernel(self):
        return eval(str(self))

    @staticmethod
    def bs(str_expr):
        return '(' + str_expr + ')'


class SumOrProductKE(KernelExpression): # Abstract
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None, symbol = '+'):
        super().__init__(root, parent)
        self.base_terms = base_terms if isinstance(base_terms, Counter) else Counter(base_terms)
        self.composite_terms = composite_terms
        self.symbol = symbol
        self.simplify_base_terms()

    def __str__(self):
        return (' ' + self.symbol + ' ').join([self.bracket_if_needed(f) for f in list(self.base_terms.elements()) + self.composite_terms])

    @staticmethod
    def bracket_if_needed(kex):
        return str(kex)

    def simplify(self):
        self.simplify_base_terms()
        self.composite_terms = [ca.simplify() for ca in self.composite_terms]
        return self

    @abstractmethod
    def simplify_base_terms(self):
        pass

    def traverse(self):
        return [self] + list(chain.from_iterable([ca.traverse() for ca in self.composite_terms]))

    def reduce(self, func, acc):
        return reduce(lambda acc2, ct: ct.reduce(func, acc2), self.composite_terms, func(self, acc))

    def set_root(self, new_root):
        self.root = new_root
        for cf in self.composite_terms: cf.set_root(new_root)
        return self

    def _set_all_parents(self):
        for ct in self.composite_terms:
            ct.parent = self
            ct._set_all_parents()
        return self

    def _check_all_parents(self):
        return all([ct.parent == self and ct._check_all_parents() for ct in self.composite_terms])

    def reassign_child(self, old_child, new_child):
        self.composite_terms.remove(old_child)
        self.composite_terms.append(new_child)
            
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
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '+')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1


class ProductKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '*')

    @staticmethod
    def bracket_if_needed(kex):
        return KernelExpression.bs(str(kex)) if isinstance(kex, SumKE) else str(kex)

    def simplify_base_terms(self):
        if self.base_terms['WN'] > 0: # WN acts as a multiplicative zero for all stationary kernels, i.e. all but LIN
            self.base_terms = Counter({'WN': 1, 'LIN': self.base_terms['LIN']})
        else:
            if self.base_terms['C'] > 0 and sum(self.base_terms.values()) - self.base_terms['C'] != 0: # If C is not the only present kernel
                self.base_terms['C'] = 0 # C is the multiplication-identity element
            elif self.base_terms['C'] > 1: self.base_terms['C'] = 1 # C is multiplication-idempotent
            if self.base_terms['SE'] > 1: self.base_terms['SE'] = 1 # SE is multiplication-idempotent


class ChangeKE(KernelExpression):
    def __init__(self, CP_or_CW, left, right, root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(root, parent)
        self.CP_or_CW = CP_or_CW
        self.left = left
        self.right = right

    def __str__(self):
        return self.CP_or_CW + KernelExpression.bs(str(self.left) + ', ' + str(self.right))

    def simplify(self):
        if isinstance(self.left, KernelExpression): self.left.simplify()
        if isinstance(self.right, KernelExpression): self.right.simplify()
        return self

    def traverse(self):
        res = [self]
        # NOTE: this version does not add new elements for raw string leaves; replace by comments for that behaviour
        res += self.left.traverse() if isinstance(self.left, KernelExpression) else []#[self.left]
        res += self.right.traverse() if isinstance(self.right, KernelExpression) else []#[self.right]
        return res

    def reduce(self, func, acc):
        # NOTE: this function does not deal with raw string leaves; see further comments upstream
        return reduce(lambda acc2, branch: branch.reduce(func, acc2),
               [branch for branch in (self.left, self.right) if isinstance(branch, KernelExpression)],
               func(self, acc))

    def set_root(self, new_root):
        self.root = new_root
        if isinstance(self.left, KernelExpression): self.left.set_root(new_root)
        if isinstance(self.right, KernelExpression): self.right.set_root(new_root)
        return self

    def _set_all_parents(self):
        if isinstance(self.left, KernelExpression):
            self.left.parent = self
            self.left._set_all_parents()
        if isinstance(self.right, KernelExpression):
            self.right.parent = self
            self.right._set_all_parents()
        return self

    def _check_all_parents(self):
        return all([ct.parent == self and ct._check_all_parents() for ct in [self.left, self.right] if isinstance(ct, KernelExpression)])

    def reassign_child(self, old_child, new_child):
        if self.left == old_child:
            self.left = new_child
        else: # elif self.right == old_child
            self.right = new_child
