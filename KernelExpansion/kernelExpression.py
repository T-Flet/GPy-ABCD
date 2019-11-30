from abc import ABC, abstractmethod
from Kernels.baseKernels import *
from Kernels.kernelOperations import *
from collections import Counter, defaultdict
from itertools import chain
from functools import reduce
import operator
from copy import deepcopy
import re


# IDEA:
#   - Models are represented by a tree of KernelExpressions, which can be Sum, Product and Change nodes;
#       each type of node may contain raw strings base kernels or composite ones, however bare raw string leaves may only occur in ChangeKEs.
#   - The tree root is stored in each node, as is the direct parent, therefore from any traversal one can just deepcopy
#       single nodes to get the rest of the tree for free.
#   - When expanding a tree: can apply all production rules at internal nodes, but only base-kernel ones at the leaves;
#       the mult/sum of sum/mult rules are applied to leaves FROM their parents; this covers the whole tree.
#   - Both traverse and reduce ignore bare string leaves and assume the user handles them from their ChangeKE parent.
#   - Methods with input arguments make deepcopies of them in order to prevent unintended modification (exceptions for methods used other methods which do)


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
    def __eq__(self, other): ## NOTE: this is intended to check equality of data fields only, i.e. it does not check root or parent
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def extract_if_singleton(self):
        pass

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
    def match_up_fit_parameters(self, fit_ker, prefix): # Note: the prefix has to already contain this node's name at the end
        pass

    @abstractmethod
    def sum_of_prods_form(self): # Return either a ProductKE or a SumKE whose composite_terms are only ProductKEs
        pass


class SumOrProductKE(KernelExpression): # Abstract
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None, symbol = None, GPy_name = None):
        super().__init__(root, parent, GPy_name)
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

    @abstractmethod
    def to_kernel(self):
        pass

    # Methods for after fit

    def match_up_fit_parameters(self, param_dict, prefix = ''):
        if self.is_root(): prefix = self.GPy_name
        elif prefix == '': raise ValueError('No prefix but not root node in match_up_fit_parameters')
        seen_terms = Counter([])
        for bt in list(self.base_terms.elements()):
            seen_terms.update([bt])
            postfix = '_' + str(seen_terms[bt] - 1) if seen_terms[bt] > 1 else ''
            self.parameters[bt].append({ p: param_dict[p_full] for p in base_k_param_names[bt]['parameters']
                                         for p_full in ['.'.join([prefix, base_k_param_names[bt]['name'] + postfix, p])] # Clunky 'let' assignment; for Python 3.8+: 'if (p_full := '.'.join([prefix, base_k_param_names[bt]['name'], p]))'
                                         if not (p == 'variance' and p_full not in param_dict) }) # I.e. skip variances if absent
        for ct in self.composite_terms:
            seen_terms.update([ct.GPy_name])
            postfix = '_' + str(seen_terms[ct.GPy_name] - 1) if seen_terms[ct.GPy_name] > 1 else ''
            ct.match_up_fit_parameters(param_dict, '.'.join([prefix, ct.GPy_name + postfix]))
        return self


class SumKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '+', 'sum')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1
        return self

    def to_kernel(self):
        return reduce(operator.add, [base_str_to_ker(bt) for bt in list(self.base_terms.elements())] + [ct.to_kernel() for ct in self.composite_terms])

    def sum_of_prods_form(self):
        return self


class ProductKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '*', 'mul')

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

    def to_kernel(self):
        bt_kers = [base_str_to_ker(bt) for bt in list(self.base_terms.elements())]
        if len(bt_kers) > 1: # I.e. leave only one of the removable variance parameters (i.e. the base_terms') per product, preferring the first factor to have it
            for btk in bt_kers[1:]: btk.unlink_parameter(btk.variance)
        return reduce(operator.mul, bt_kers + [ct.to_kernel() for ct in self.composite_terms])

    # Methods for after fit

    def match_up_fit_parameters(self, param_dict, prefix = ''):
        super().match_up_fit_parameters(param_dict, prefix)
        if sum(self.base_terms.values()) > 0:
            single_variance = None
            for k, v_list in self.parameters.items():
                for i in range(len(v_list)):
                    if 'variance' in v_list[i]: # Extract the one variance left in the base_terms from its original term and assign it to the ProductKE itself
                        single_variance = v_list[i]['variance']
                        del v_list[i]['variance']
                        break
                if single_variance is not None: break
            self.parameters['ProductKE'].append({'variance': single_variance})
        return self

    def new_bases_with_parameters(self, base_parameters): # tuple or list of tuples
        bpss = [base_parameters] if isinstance(base_parameters, tuple) else base_parameters
        for b, ps in bpss:
            self.new_base(b)
            self.parameters[b].append(ps)

        ## SIMPLIFY WITH PARAMETERS HERE (e.g. remove all others if WN present)!!!!!!!!!!!!!!!!!

        return self

    def sum_of_prods_form(self):
        return self


from KernelExpansion.kernelExpressionOperations import add_sum_of_prods_terms # Needs to be here; requires SumKE and ProductKE

class ChangeKE(KernelExpression):
    def __init__(self, CP_or_CW, left, right, root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(root, parent, base_k_param_names[CP_or_CW]['name'])
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
        if self.left is old_child: self.left = new_child # NOT A deepcopy!
        else: self.right = new_child # NOT A deepcopy! # I.e. elif self.right is old_child
        return new_child # NOTE THIS RETURN VALUE (used by new_tree_with_self_replaced)

    def to_kernel(self):
        left_ker = self.left.to_kernel() if isinstance(self.left, KernelExpression) else base_str_to_ker(self.left)
        right_ker = self.right.to_kernel() if isinstance(self.right, KernelExpression) else base_str_to_ker(self.right)
        return base_str_to_ker_func[self.CP_or_CW](left_ker, right_ker)

    # Methods for after fit

    def match_up_fit_parameters(self, param_dict, prefix = ''):
        if self.is_root(): new_prefix = prefix = self.GPy_name
        elif prefix == '': raise ValueError('No prefix but not root node in match_up_fit_parameters')
        self.parameters[self.CP_or_CW].append({p: param_dict['.'.join([prefix, p])] for p in base_k_param_names[self.CP_or_CW]['parameters']})
        same_type_branches = type(self.left) == type(self.right) and\
                             ((isinstance(self.left, KernelExpression) and self.left.GPy_name == self.right.GPy_name) or self.left == self.right)
        for child in (('left', self.left), ('right', self.right)):
            postfix = '_1' if same_type_branches and child[0] == 'right' else ''
            if isinstance(child[1], KernelExpression): child[1].match_up_fit_parameters(param_dict, '.'.join([prefix, child[1].GPy_name + postfix]))
            else: self.parameters[child].append({p: param_dict['.'.join([prefix, base_k_param_names[child[1]]['name'] + postfix, p])] for p in base_k_param_names[child[1]]['parameters']})
        return self

    def sum_of_prods_form(self):
        new_children = []
        for child in (('left', self.left), ('right', self.right)):
            sigmoid_parameters = (change_k_sigmoid_names[self.CP_or_CW][child[0]], self.parameters[self.CP_or_CW])
            if isinstance(child[1], str):
                leaf_params = [self.parameters[k][0] for k in self.parameters.keys() if isinstance(k, tuple) and k[0] == child[0]][0]
                new_children.append(ProductKE([]).new_bases_with_parameters([(child[1], leaf_params), sigmoid_parameters]))
            else:
                new_child = child[1].sum_of_prods_form()
                if isinstance(new_child, ProductKE): new_child.new_bases_with_parameters(sigmoid_parameters)
                else: # I.e. SumKE
                    for pt in new_child.composite_terms: pt.new_bases_with_parameters(sigmoid_parameters)
                    for bt in new_child.base_terms.elements():
                        match_ps = new_child.parameters[bt]
                        new_child.new_composite(ProductKE([]).new_bases_with_parameters([(bt, match_ps[0]), sigmoid_parameters]))
                        if len(match_ps) == 1: new_child.parameters.pop(bt)
                        else: del match_ps[0]
                    new_child.base_terms.clear()
                new_children.append(new_child)
        return add_sum_of_prods_terms(new_children[0], new_children[1]).set_parent(self.parent).set_root(self.root)



# TODO:
#   - Redo all ChangeKE methods using 'for branch in (self.left, self.right):' instead of repeated code where appropriate
