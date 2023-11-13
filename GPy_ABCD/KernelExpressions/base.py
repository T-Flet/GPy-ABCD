import re
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict

from GPy_ABCD.Kernels.baseKernels import * # Here only for the .to_kernel_unrefined method evaluation context
from GPy_ABCD.KernelExpansion.kernelInterpretation import base_factors_interpretation


# IDEA:
#   - Models are represented by a tree of KernelExpressions, which can be Sum, Product and Change nodes;
#       each type of node may contain raw strings base kernels or composite ones, however bare raw string leaves may only occur in ChangeKEs.
#   - The tree root is stored in each node, as is the direct parent, therefore from any traversal one can just deepcopy
#       single nodes to get the rest of the tree for free.
#   - When expanding a tree: can apply all production rules at internal nodes, but only base-kernel ones at the leaves;
#       the mult/sum of sum/mult rules are applied to leaves FROM their parents; this covers the whole tree.
#   - Both traverse and reduce ignore bare string leaves and assume the user handles them from their ChangeKE parent.
#   - Methods with input arguments make deepcopies of them in order to prevent unintended modification (exceptions for methods used other methods which do)


class KernelExpression(ABC): # Abstract
    def __init__(self, root, parent, GPy_name = None):
        super().__init__()
        self.root = root
        self.parent = parent
        self.GPy_name = GPy_name
        self.parameters = defaultdict(list)

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
    def __repr__(self):
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

    @abstractmethod
    def traverse(self):
        '''NOTE: both traverse and reduce ignore raw-string leaves (which can only happen in ChangeKEs);
        care has to be taken to perform required operations on them from their parent'''
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
    def reassign_child(self, old_child, new_child):
        '''NOTE: has to return new_child (used by new_tree_with_self_replaced)'''
        pass

    @abstractmethod
    def contains_base(self, bt):
        pass

    @abstractmethod
    def is_stationary(self):
        pass

    @abstractmethod
    def to_kernel(self):
        pass

    def to_kernel_unrefined(self):
        # return eval(str(self))
        return eval(re.sub(r'([A-Z]+(?!\(|[A-Z]))', r'\1()', str(self))) # Call the kernels in case they are functions returning a new instance

    @staticmethod
    def bs(str_expr):
        return '(' + str_expr + ')'

    # Methods for after fit

    def _new_parameters(self, new_parameters):
        self.parameters.update(new_parameters)
        return self

    @abstractmethod
    def match_up_fit_parameters(self, fit_ker, prefix):
        '''NOTE: the prefix has to already contain THIS node's name followed by a dot at the end'''
        pass

    @abstractmethod
    def sum_of_prods_form(self):
        '''Return either a ProductKE or a SumKE whose composite_terms are only ProductKEs.

        NOTE: this method CAN only be called when parameters are present (i.e. after .match_up_fit_parameters has been called),
        and SHOULD only be called indirectly through GPModel.sum_of_prods_kex or GPModel.interpret()'''
        pass

    def get_interpretation(self, sops = None):
        if sops is None: sops = self.sum_of_prods_form()
        assert sum(sops.base_terms.values()) == 0, f'Some base terms are left in the expanded sum of products form: {sops.base_terms.values()}'
        description = f'The fit kernel consists of {len(sops.composite_terms)} components:'
        for ct in sops.composite_terms: description += '\n\t' + base_factors_interpretation(ct.parameters)
        return description


