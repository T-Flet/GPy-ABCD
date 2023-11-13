import operator

from GPy_ABCD.KernelExpressions.base import KernelExpression
from GPy_ABCD.KernelExpressions.commutatives import SumKE, ProductKE
from GPy_ABCD.KernelExpansion.kernelOperations import *
from GPy_ABCD.KernelExpansion.kernelInterpretation import *
from GPy_ABCD.Util.genericUtil import update_dict_with
from GPy_ABCD.Util.kernelUtil import sortOutTypePair


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
        # NOTE: this version adds new elements for raw string leaves (making them SumKE singletons); replace by comments to remove that behaviour
        res += self.left.traverse() if isinstance(self.left, KernelExpression) else [SumKE([self.left]).set_root(self.root).set_parent(self)]#[]
        res += self.right.traverse() if isinstance(self.right, KernelExpression) else [SumKE([self.right]).set_root(self.root).set_parent(self)]#[]
        return res

    def reduce(self, func, acc):
        # NOTE: this function DOES deal with raw string leaves; see further comments upstream; swap commented middle line for opposite behaviour
        return reduce(lambda acc2, branch: branch.reduce(func, acc2),
               [branch if isinstance(branch, KernelExpression) else SumKE([branch]).set_root(self.root).set_parent(self) for branch in (self.left, self.right)],
               # [branch for branch in (self.left, self.right) if isinstance(branch, KernelExpression)],
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

    def contains_base(self, bts):
        if not isinstance(bts, list): bts = [bts]
        return any([branch in bts if isinstance(branch, str) else branch.contains_base(bts) for branch in (self.left, self.right)])

    def is_stationary(self):
        return False

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
        for branch, kex in (('left', self.left), ('right', self.right)):
            postfix = '_1.' if same_type_branches and branch == 'right' else '.'
            if isinstance(kex, KernelExpression): kex.match_up_fit_parameters(param_dict, prefix + kex.GPy_name + postfix)
            else: self.parameters[(branch, kex)].append({p: param_dict[prefix + base_k_param_names[kex]['name'] + postfix + p] for p in base_k_param_names[kex]['parameters']})
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
        assert self.parameters, 'A sum-of-products form can only be generated when parameters are present (i.e. after .match_up_fit_parameters has been triggered), and should only be called indirectly through GPModel.sum_of_prods_kex or GPModel.interpret()'
        new_children = []
        for branch, kex in (('left', self.left), ('right', self.right)):
            sigmoid_parameters = (change_k_sigmoid_names[self.CP_or_CW][branch], self.parameters[self.CP_or_CW][0])
            if isinstance(kex, str):
                leaf_params = [self.parameters[k][0] for k in self.parameters.keys() if isinstance(k, tuple) and k[0] == branch][0]
                new_children.append(ProductKE([]).new_bases_with_parameters([(kex, leaf_params), sigmoid_parameters]))
            else:
                new_child = kex.sum_of_prods_form()
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


