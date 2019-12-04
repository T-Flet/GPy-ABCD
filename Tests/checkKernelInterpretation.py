from KernelExpansion.kernelExpression import *
from KernelExpansion.kernelExpressionOperations import *
from KernelExpansion.kernelInterpretation import *


# # testExpr = init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['C', 'PER'])]))
# # testExpr = init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'WN', 'PER'), SumKE(['C', 'PER'])]))
# testExpr = init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])]))
# res = testExpr.sum_of_prods_form()
# print(res)
# print(res.parameters)
# print(res.composite_terms[0].parameters)
#
# component_n = 2
# del res.composite_terms[component_n].parameters['ProductKE']
# ordered_ps = sorted(res.composite_terms[component_n].parameters.items(), key = lambda bps: base_kern_interp_order[bps[0]])
#
# res = first_term_interpretation(ordered_ps[0])
# print(res)



## Sigmoid overlaps

# s0 = {'location': 0, 'slope': 0}
# s1 = {'location': 1, 'slope': 1}
# s2 = {'location': 2, 'slope': 2}
# res = S_overlap([s0, s1, s2])

# sr0 = {'location': 0, 'slope': 0}
# sr1 = {'location': 1, 'slope': 1}
# sr2 = {'location': 2, 'slope': 2}
# res = Sr_overlap([sr0, sr1, sr2])

# si0 = {'location': 0, 'slope': 0, 'width': 2} # +-1
# si1 = {'location': 1, 'slope': 1, 'width': 1} # 0.5, 1.5
# si2 = {'location': 0.25, 'slope': 2, 'width': 1} # -0.25, 0.75
# res = SI_overlap([si0, si1, si2])

# sir0 = {'location': 0, 'slope': 0, 'width': 2} # +-1
# sir1 = {'location': 1, 'slope': 1, 'width': 1} # 0.5, 1.5
# sir2 = {'location': 0.25, 'slope': 2, 'width': 1} # -0.25, 0.75
# res = SIr_overlap([sir0, sir1, sir2])


# print(res)