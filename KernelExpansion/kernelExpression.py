from abc import ABC, abstractmethod
from Kernels.baseKernels import *
from collections import Counter
from itertools import chain
from functools import reduce
from copy import deepcopy
# import kernelExpressionOperations


# IDEA:
#   - Models are represented by a tree of KernelExpressions, which can be Sum, Product and Change nodes;
#       each type of node may contain raw strings base kernels or composite ones, however bare raw string leaves may only occur in ChangeKEs.
#   - The tree root is stored in each node, as is the direct parent, therefore from any traversal one can just deepcopy
#       single nodes to get the rest of the tree for free.
#   - When expanding a tree: can apply all production rules at internal nodes, but only base-kernel ones at the leaves;
#       the mult/sum of sum/mult rules are applied to leaves FROM their parents; this covers the whole tree.
#   - Both traverse and reduce ignore bare string leaves and assume the user handles them from their ChangeKE parent.
#   - Methods with input arguments make deepcopies of them in order to prevent unintended modification (exceptions for methods used other methods which do)


# TODO:
#   Think about the issue of accumulating variance parameters in repeated multiplications GPy.kern multiplications; worth trying to prevent it?


def lists_of_unhashables__eq(xs, ys):
    cys = list(ys) # make a mutable copy
    try:
        for x in xs: cys.remove(x)
    except ValueError: return False
    return not cys

def lists_of_unhashables__diff(xs, ys):
    cxs = list(xs) # make a mutable copy
    try:
        for y in ys: cxs.remove(y)
    except ValueError: pass
    return cxs


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
    def __eq__(self, other): ## NOTE: this is intended to check equality of data fields only, i.e. it does not check root or parent
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def extract_if_singleton(self):
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

    def is_root(self):
        if self.root is self:
            assert self.parent is None, 'Something went rong: ' + str(self) + ' is the root while its parent is not None but ' + str(self.parent)
            return True
        else: return False

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

    def _initialise(self): # NOTE: The argument is intended to be the root of a manually coded tree
        return self.set_root()._set_all_parents().simplify()

    def new_tree_with_self_replaced(self, replacement_node): # NOTE: replacement_node is assumed to already have handled changing its childrens' parent to itself
        copied_replacement, copied_self = deepcopy((replacement_node, self))
        if copied_self.is_root():
            return copied_replacement.set_parent(None).set_root()
        else:
            copied_replacement.set_parent(copied_self.parent).set_root(copied_self.root)
            return copied_replacement.parent.reassign_child(copied_self, copied_replacement)

    @abstractmethod
    def reassign_child(self, old_child, new_child): # NOTE: has to return new_child (used by new_tree_with_self_replaced)
        pass

    def to_kernel(self):
        return eval(str(self))

    @staticmethod
    def bs(str_expr):
        return '(' + str_expr + ')'


class SumOrProductKE(KernelExpression): # Abstract
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None, symbol = '+'):
        super().__init__(root, parent)
        self.base_terms = deepcopy(base_terms) if isinstance(base_terms, Counter) else Counter(base_terms)
        self.composite_terms = deepcopy(composite_terms)
        self.symbol = symbol
        # self.simplify_base_terms() # Activate only this instead of full simplify for some testing
        self.simplify()
        for ct in self.composite_terms: ct.set_parent(self).set_root(self.root)

    def __str__(self):
        return (' ' + self.symbol + ' ').join([self.bracket_if_needed(f) for f in list(self.base_terms.elements()) + self.composite_terms])

    def __eq__(self, other): ## NOTE: this is intended to check equality of data fields only, i.e. it does not check root or parent
        return type(self) == type(other) and self.base_terms == other.base_terms and lists_of_unhashables__eq(self.composite_terms, other.composite_terms)

    @staticmethod
    def bracket_if_needed(kex):
        return str(kex)

    def simplify(self):
        self.composite_terms = [ct.simplify() for ct in self.composite_terms]
        return self.absorb_homogeneous_composites().absorb_singletons().simplify_base_terms()

    @abstractmethod
    def simplify_base_terms(self):
        pass

    def absorb_singletons(self):
        singletons = filter(lambda x: isinstance(x[1], str), [(ct, ct.extract_if_singleton()) for ct in self.composite_terms])
        for s in singletons:
            self.new_base(s[1])
            self.composite_terms.remove(s[0])
        return self

    def extract_if_singleton(self): # This modifies the composite_child's parent if that kind of singleton
        if sum(self.base_terms.values()) == 1 and len(self.composite_terms) == 0: return list(self.base_terms.elements())[0]
        elif sum(self.base_terms.values()) == 0 and len(self.composite_terms) == 1: return self.composite_terms[0].set_parent(self.parent)
        else: return self

    def absorb_homogeneous_composites(self):
        homogeneous_composites = [ct for ct in self.composite_terms if isinstance(ct, type(self))] # Annoyingly less problematic than lambda-using-self filter
        for hc in homogeneous_composites:
            self.base_terms.update(hc.base_terms)
            self.simplify_base_terms()
            for hcct in hc.composite_terms: hcct.parent = self
            self.composite_terms += hc.composite_terms
            self.composite_terms.remove(hc)
        return self

    def traverse(self):
        return [self] + list(chain.from_iterable([ct.traverse() for ct in self.composite_terms]))

    def reduce(self, func, acc):
        return reduce(lambda acc2, ct: ct.reduce(func, acc2), self.composite_terms, func(self, acc))

    def set_root(self, new_root = None):
        if new_root is None: new_root = self
        self.root = new_root
        for ct in self.composite_terms: ct.set_root(new_root)
        return self

    def _set_all_parents(self):
        for ct in self.composite_terms:
            ct.parent = self
            ct._set_all_parents()
        return self

    def _check_all_parents(self):
        return all([ct.parent is self and ct._check_all_parents() for ct in self.composite_terms])

    def reassign_child(self, old_child, new_child):
        self.composite_terms.remove(old_child)
        self.composite_terms.append(new_child) # NOT A deepcopy!
        return new_child # NOTE THIS RETURN VALUE (used by new_tree_with_self_replaced)
            
    def new_base(self, new_base_terms):
        if isinstance(new_base_terms, str):
            self.base_terms[new_base_terms] += 1
        else: # list or Counter
            self.base_terms.update(new_base_terms)
        self.simplify_base_terms()
        return self

    def new_composite(self, new_composite_terms):
        if isinstance(new_composite_terms, KernelExpression):
            self.composite_terms += [deepcopy(new_composite_terms).set_parent(self).set_root(self.root)]
        else: # list
            self.composite_terms += [deepcopy(nct).set_parent(self).set_root(self.root) for nct in new_composite_terms]
        return self

    def term_count(self):
        return sum(self.base_terms.values()) + len(self.composite_terms)


class SumKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '+')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1
        return self


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
            if self.base_terms['C'] > 0: # C is the multiplication-identity element, therefore remove it unless it is the only factor
                self.base_terms['C'] = 0 if self.term_count() != self.base_terms['C'] else 1
            if self.base_terms['SE'] > 1: self.base_terms['SE'] = 1 # SE is multiplication-idempotent
        return self


class ChangeKE(KernelExpression):
    def __init__(self, CP_or_CW, left, right, root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(root, parent)
        self.CP_or_CW = CP_or_CW
        self.left = deepcopy(left)
        self.right = deepcopy(right)
        self.simplify() # Deactivate for some testing
        if isinstance(self.left, KernelExpression): self.left.set_parent(self).set_root(self.root)
        if isinstance(self.right, KernelExpression): self.right.set_parent(self).set_root(self.root)

    def __str__(self):
        return self.CP_or_CW + KernelExpression.bs(str(self.left) + ', ' + str(self.right))

    def __eq__(self, other): ## NOTE: this is intended to check equality of data fields only, i.e. it does not check root or parent
        return type(self) == type(other) and self.CP_or_CW == other.CP_or_CW and self.left == other.left and self.right == other.right

    def simplify(self):
        if isinstance(self.left, KernelExpression): self.left = self.left.simplify().extract_if_singleton()
        if isinstance(self.right, KernelExpression): self.right = self.right.simplify().extract_if_singleton()
        return self

    def extract_if_singleton(self):
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

    def set_root(self, new_root = None):
        if new_root is None: new_root = self
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
        return all([ct.parent is self and ct._check_all_parents() for ct in [self.left, self.right] if isinstance(ct, KernelExpression)])

    def reassign_child(self, old_child, new_child):
        if self.left is old_child:
            self.left = new_child # NOT A deepcopy!
        else: # elif self.right is old_child
            self.right = new_child # NOT A deepcopy!
        return new_child # NOTE THIS RETURN VALUE (used by new_tree_with_self_replaced)
