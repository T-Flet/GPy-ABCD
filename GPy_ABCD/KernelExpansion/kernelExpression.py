import operator
import re
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import chain, product

from GPy_ABCD.KernelExpansion.kernelOperations import *
from GPy_ABCD.KernelExpansion.kernelInterpretation import *
from GPy_ABCD.Util.genericUtil import sortOutTypePair, update_dict_with, partition, lists_of_unhashables__eq


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
    def match_up_fit_parameters(self, fit_ker, prefix): # Note: the prefix has to already contain THIS node's name followed by a dot at the end
        pass

    @abstractmethod
    def sum_of_prods_form(self): # Return either a ProductKE or a SumKE whose composite_terms are only ProductKEs
        pass

    def get_interpretation(self, sops = None):
        if sops is None: sops = self.sum_of_prods_form()
        assert sum(sops.base_terms.values()) == 0, f'Some base terms are left in the expanded sum of products form: {sops.base_terms.values()}'
        description = f'The fit kernel consists of {len(sops.composite_terms)} components:'
        for ct in sops.composite_terms: description += '\n\t' + base_factors_interpretation(ct.parameters)
        return description


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
        return (' ' + self.symbol + ' ').join([self.bracket_if_needed(f) for f in order_base_kerns(list(self.base_terms.elements())) + self.composite_terms])

    def __repr__(self):
        res = type(self).__name__ + '([' + ', '.join(["'"+bt+"'" for bt in self.base_terms.elements()]) + ']'
        cts = ', [' + ', '.join([ct.__repr__() for ct in self.composite_terms]) + ']' if self.composite_terms else ''
        return res + cts + ')'

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
        (bts, self.composite_terms) = partition(lambda x: isinstance(x, str), [ct.extract_if_singleton() for ct in self.composite_terms])
        self.new_base(bts)
        return self

    def _is_singleton(self):
        return sum(self.base_terms.values()) + len(self.composite_terms) == 1

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
        if self.is_root(): prefix += '' if self._is_singleton() else self.GPy_name + '.'
        elif prefix == '': raise ValueError('No prefix but not root node in match_up_fit_parameters')
        seen_terms = Counter([])
        for bt in list(self.base_terms.elements()):
            seen_terms.update([bt])
            postfix = '_' + str(seen_terms[bt] - 1) + '.' if seen_terms[bt] > 1 else '.'
            self.parameters[bt].append({ p: param_dict[p_full] for p in base_k_param_names[bt]['parameters']
                                         for p_full in [prefix + base_k_param_names[bt]['name'] + postfix + p] # Clunky 'let' assignment; for Python 3.8+: 'if (p_full := '.'.join([prefix, base_k_param_names[bt]['name'], p]))'
                                         if not (p == 'variance' and p_full not in param_dict) }) # I.e. skip variances if absent
        for ct in self.composite_terms:
            seen_terms.update([ct.GPy_name])
            postfix = '_' + str(seen_terms[ct.GPy_name] - 1) + '.' if seen_terms[ct.GPy_name] > 1 else '.'
            ct.match_up_fit_parameters(param_dict, prefix + ct.GPy_name + postfix)
        return self

    @abstractmethod
    def sum_of_prods_form(self):
        pass


class SumKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '+', 'sum')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1
        self.base_terms = + self.base_terms
        return self

    def to_kernel(self):
        return reduce(operator.add, [base_str_to_ker(bt) for bt in order_base_kerns(list(self.base_terms.elements()))] + [ct.to_kernel() for ct in self.composite_terms])

    # Methods for after fit

    def simplify_base_terms_params(self):
        if 'WN' in self.parameters and len(self.parameters['WN']) > 1:
            self.parameters['WN'] = [{'variance': reduce(operator.add, [ps['variance'] for ps in self.parameters['WN']])}]
        if 'C' in self.parameters and len(self.parameters['C']) > 1:
            self.parameters['C'] = [{'variance': reduce(operator.add, [ps['variance'] for ps in self.parameters['C']])}]
        return self

    def sum_of_prods_form(self):
        cts = [ct.sum_of_prods_form() for ct in self.composite_terms]
        self.composite_terms.clear()
        for ct in cts: # Only SumKEs or ProductKEs now
            if isinstance(ct, SumKE): # The only other type left at this stage is ProductKE, and no change is required for it
                assert sum(ct.base_terms.values()) == 0, 'There should not be any base terms in nested SumKEs coming from nested sum_of_prods_form'
                # self.new_base(ct.base_terms)
                # self._new_parameters(ct.parameters)
                # self.simplify_base_terms_params()
                self.new_composite(ct.composite_terms)
            else: self.composite_terms.append(ct)

        for bt in self.base_terms.elements(): # Move all base_terms to composite_terms as singleton ProductKEs (prepending them to the existing composites)
            self.composite_terms.insert(0, ProductKE([]).new_bases_with_parameters((bt, self.parameters[bt][0])).set_parent(self).set_root(self.root))
            self.base_terms[bt] -= 1
            if len(self.parameters[bt]) == 1: self.parameters.pop(bt)
            else: del self.parameters[bt][0]
        assert sum(self.base_terms.values()) == 0 and len(self.parameters) == 0, 'There are some remaining base_terms or parameters after trying to port them to composite ones'
        self.base_terms = +self.base_terms  # Clear 0s for neatness

        assert all([isinstance(ct, ProductKE) for ct in self.composite_terms]), 'Some composite_terms of a SumKE after sum_of_prods_form are not-ProductKEs'
        return self


class ProductKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '*', 'mul')

    @staticmethod
    def bracket_if_needed(kex):
        return KernelExpression.bs(str(kex)) if isinstance(kex, SumKE) else str(kex)

    def simplify_base_terms(self):
        if self.base_terms['WN'] > 0: # WN acts as a multiplicative zero for all stationary kernels, i.e. all but LIN and sigmoids
            for bt in list(self.base_terms.keys()):
                if bt not in ['WN', 'LIN'] + list(base_sigmoids): del self.base_terms[bt]
        else:
            if self.base_terms['C'] > 0: # C is the multiplication-identity element, therefore remove it unless it is the only factor
                self.base_terms['C'] = 0 if self.term_count() != self.base_terms['C'] else 1
            if self.base_terms['SE'] > 1: self.base_terms['SE'] = 1 # SE is multiplication-idempotent
        self.base_terms = + self.base_terms
        return self

    def to_kernel(self):
        bt_kers = [base_str_to_ker(bt) for bt in order_base_kerns(list(self.base_terms.elements()))]
        if len(bt_kers) > 1: # I.e. leave only one of the removable variance parameters (i.e. the base_terms') per product, preferring the first factor to have it
            for btk in bt_kers[1:]: btk.unlink_parameter(btk.variance)
        return reduce(operator.mul, bt_kers + [ct.to_kernel() for ct in self.composite_terms])

    # Methods for after fit

    def simplify_base_terms_params(self):
        if 'WN' in self.parameters:
            for bt in list(self.parameters.keys()):
                if bt not in ['WN', 'ProductKE'] + list(base_sigmoids): del self.parameters[bt]
        if 'C' in self.parameters:
            if 'C' not in self.base_terms: del self.parameters['C']
            elif len(self.parameters['C']) > 1: self.parameters['C'] = [dict()]
        if 'SE' in self.parameters and len(self.parameters['SE']) > 1:
            self.parameters['SE'] = [{'lengthscale': reduce(lambda acc, l: (acc + l) / np.sqrt(acc ** 2 + l ** 2), [ps['lengthscale'] for ps in self.parameters['SE']])}]
        return self

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
        if 'ProductKE' not in self.parameters: self.parameters['ProductKE'] = [{'variance': 1.}]
        bpss = [base_parameters] if isinstance(base_parameters, tuple) else base_parameters
        for b, ps in bpss:
            if 'variance' in ps:
                self.parameters['ProductKE'][0]['variance'] *= ps['variance']
                if b == 'ProductKE': continue
                else: del ps['variance']
            self.base_terms[b] += 1 # Not using new_base since have a simplify_base_terms later
            self.parameters[b].append(ps)
        self.simplify_base_terms()
        self.simplify_base_terms_params()
        return self

    @staticmethod # This would live in kernelExpressionOperations if it did not need to be used within ProductKEs; not changing k0 to self and keeping it static in that spirit
    def multiply_pure_prods_with_params(k0, ks): # k0 is meant to be the ProductKE containing the pure (i.e. base_terms-only) ks
        assert sum([len(kex.composite_terms) for kex in ks]) == 0, 'Arguments (not k0) of multiply_pure_prods_with_params are not base_terms-only'
        return ProductKE([]).new_bases_with_parameters([(key, p) for kex in [k0] + ks for key, ps in list(kex.parameters.items()) for p in ps])

    def sum_of_prods_form(self):
        sops = SumKE([])
        if not self.composite_terms:
            sops.composite_terms.append(ProductKE(self.base_terms)._new_parameters(self.parameters)) # Avoid triggering simplify()
            return sops.set_parent(self.parent)._set_all_parents().set_root(self.root)
        else:
            self.composite_terms = [ct.sum_of_prods_form() for ct in self.composite_terms]
            assert all([isinstance(ct, SumKE) for ct in self.composite_terms]), 'Some non-SumKE terms are coming from sum_of_prods_form calls within a ProductKE composite_terms'
            sets_of_factors = product(*[cts.composite_terms for cts in self.composite_terms]) # Cartesian product of all sums of products
            sops.composite_terms = [self.multiply_pure_prods_with_params(self, list(factor_tuple)) for factor_tuple in sets_of_factors] # Avoid triggering simplify() on these expanded composites product
            return sops.set_parent(self.parent)._set_all_parents().set_root(self.root)


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

    def __repr__(self):
        return f"{type(self).__name__}('{self.CP_or_CW}', {self.left.__repr__()}, {self.right.__repr__()})"

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

    def to_kernel_with_params(self): # To be used on the result of a sum_of_prods_form
        assert True, 'to_kernel_with_params called on a ChangeKE; only SumKE and ProductKE terms should be left when calling it after sum_of_prods_form'

    def match_up_fit_parameters(self, param_dict, prefix = ''):
        if self.is_root(): prefix += self.GPy_name + '.'
        elif prefix == '': raise ValueError('No prefix but not root node in match_up_fit_parameters')
        self.parameters[self.CP_or_CW].append({p: param_dict[prefix + p] for p in base_k_param_names[self.CP_or_CW]['parameters']})
        same_type_branches = type(self.left) == type(self.right) and\
                             ((isinstance(self.left, KernelExpression) and self.left.GPy_name == self.right.GPy_name) or self.left == self.right)
        for child in (('left', self.left), ('right', self.right)):
            postfix = '_1.' if same_type_branches and child[0] == 'right' else '.'
            if isinstance(child[1], KernelExpression): child[1].match_up_fit_parameters(param_dict, prefix + child[1].GPy_name + postfix)
            else: self.parameters[child].append({p: param_dict[prefix + base_k_param_names[child[1]]['name'] + postfix + p] for p in base_k_param_names[child[1]]['parameters']})
        return self

    @staticmethod # This would live in kernelExpressionOperations if it did not need to be used within ChangeKEs
    def add_sum_of_prods_terms(k1, k2):
        res = None
        pair = sortOutTypePair(k1, k2)
        if len(pair) == 1:
            if isinstance(k1, ProductKE): res = SumKE([], [k1, k2])
            else: res = SumKE(+k1.base_terms + k2.base_terms, k1.composite_terms + k2.composite_terms)._new_parameters(update_dict_with(deepcopy(k1.parameters), k2.parameters, operator.add))
        else:  # I.e. one SumKE and one ProductKE
            if isinstance(k1, ProductKE): res = SumKE(+k2.base_terms, [k1] + k2.composite_terms)._new_parameters(k2.parameters)
            else: res = SumKE(+k1.base_terms, k1.composite_terms + [k2])._new_parameters(k1.parameters)
        return res._set_all_parents()

    def sum_of_prods_form(self):
        new_children = []
        for child in (('left', self.left), ('right', self.right)):
            sigmoid_parameters = (change_k_sigmoid_names[self.CP_or_CW][child[0]], self.parameters[self.CP_or_CW][0])
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
        return self.add_sum_of_prods_terms(new_children[0], new_children[1]).set_parent(self.parent).set_root(self.root)



# TODO:
#   - Redo all ChangeKE methods using 'for branch in (self.left, self.right):' instead of repeated code where appropriate
