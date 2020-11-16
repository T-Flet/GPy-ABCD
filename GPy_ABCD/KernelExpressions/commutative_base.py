from abc import abstractmethod
from collections import Counter
from itertools import chain

from GPy_ABCD.KernelExpressions.base import KernelExpression
from GPy_ABCD.KernelExpansion.kernelOperations import non_stationary_kerns, base_k_param_names
from GPy_ABCD.KernelExpansion.kernelInterpretation import *
from GPy_ABCD.Util.genericUtil import partition, eq_elems


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
        return type(self) == type(other) and self.base_terms == other.base_terms and eq_elems(self.composite_terms, other.composite_terms)

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

    def contains_base(self, bts):
        if not isinstance(bts, list): bts = [bts]
        return any([bt in self.base_terms for bt in bts]) or any([ct.contains_base(bts) for ct in self.composite_terms])

    def is_stationary(self):
        return all([ns not in self.base_terms for ns in non_stationary_kerns]) and all([ct.is_stationary() for ct in self.composite_terms])

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


