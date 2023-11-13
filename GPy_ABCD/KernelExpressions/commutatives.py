import operator
import numpy as np
from itertools import product

from GPy_ABCD.KernelExpressions.commutative_base import KernelExpression, SumOrProductKE
from GPy_ABCD.KernelExpansion.kernelOperations import *
from GPy_ABCD.KernelExpansion.kernelInterpretation import *
import GPy_ABCD.config as config


class SumKE(SumOrProductKE):
    def __init__(self, base_terms, composite_terms = [], root: KernelExpression = None, parent: KernelExpression = None):
        super().__init__(base_terms, composite_terms, root, parent, '+', 'sum')

    def simplify_base_terms(self):
        # WN and C are addition-idempotent
        if self.base_terms['WN'] > 1: self.base_terms['WN'] = 1
        if self.base_terms['C'] > 1: self.base_terms['C'] = 1

        # If an offset-including LIN is used, remove any C in their presence (going through __dict__ because the class name gets prepended otherwise)
        if config.__dict__['__USE_LIN_KERNEL_HORIZONTAL_OFFSET'] and self.base_terms['LIN'] > 0: self.base_terms['C'] = 0

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
        assert self.parameters, 'A sum-of-products form can only be generated when parameters are present (i.e. after .match_up_fit_parameters has been triggered), and should only be called indirectly through GPModel.sum_of_prods_kex or GPModel.interpret()'
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
            self.base_terms['WN'] = 1 # It is also idempotent
            for bt in list(self.base_terms.keys()):
                if bt not in ['WN', 'LIN'] + list(base_sigmoids): del self.base_terms[bt]
        else:
            if self.base_terms['C'] > 0: # C is the multiplication-identity element, therefore remove it unless it is the only factor or alone with a sigmoidal
                base_sigmoidals_count = sum([self.base_terms[sb] for sb in base_sigmoids if sb in self.base_terms.keys()])
                self.base_terms['C'] = 0 if self.term_count() - base_sigmoidals_count != self.base_terms['C'] else 1
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
        assert self.parameters, 'A sum-of-products form can only be generated when parameters are present (i.e. after .match_up_fit_parameters has been triggered), and should only be called indirectly through GPModel.sum_of_prods_kex or GPModel.interpret()'
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


