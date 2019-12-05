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

# s0 = S_interval({'location': 0, 'slope': 0})
# s1 = S_interval({'location': 1, 'slope': 1})
# s2 = S_interval({'location': 2, 'slope': 2})
# res_S = S_overlap([s0, s1, s2])
# print(res_S)
#
# sr0 = Sr_interval({'location': 0, 'slope': 0})
# sr1 = Sr_interval({'location': 1, 'slope': 1})
# sr2 = Sr_interval({'location': 2, 'slope': 2})
# res_Sr = Sr_overlap([sr0, sr1, sr2])
# print(res_Sr)
#
# si0 = SI_interval({'location': 0, 'slope': 0, 'width': 2}) # +-1
# si1 = SI_interval({'location': 1, 'slope': 1, 'width': 1}) # 0.5, 1.5
# si2 = SI_interval({'location': 0.25, 'slope': 2, 'width': 1}) # -0.25, 0.75
# res_SI = SI_overlap([si0, si1, si2])
# print(res_SI)
#
# sir0 = SIr_hole_interval({'location': 0, 'slope': 0, 'width': 2}) # +-1
# sir1 = SIr_hole_interval({'location': 1, 'slope': 1, 'width': 1}) # 0.5, 1.5
# sir2 = SIr_hole_interval({'location': 0.25, 'slope': 2, 'width': 1}) # -0.25, 0.75
# res_SIr = SIr_overlap([sir0, sir1, sir2])
# print(res_SIr)


# # res = simplify_sigmoidal_intervals(('S', {'start': 2, 'start_slope': 2}), ('SI', {'start': 1, 'end': 4, 'start_slope': 1, 'end_slope': 3}))
# # res = simplify_sigmoidal_intervals(('S', {'start': 2, 'start_slope': 2}), ('Sr', {'end': 4, 'end_slope': 0}))
# # res = simplify_sigmoidal_intervals(('S', {'start': 2, 'start_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
# # res = simplify_sigmoidal_intervals(('S', {'start': 0.5, 'start_slope': 2}), ('SIr', [{'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
#
# # res = simplify_sigmoidal_intervals(('Sr', {'end': 1, 'end_slope': 0}), ('SI', {'start': 0.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}))
# # res = simplify_sigmoidal_intervals(('Sr', {'end': 0, 'end_slope': 0}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
# # res = simplify_sigmoidal_intervals(('Sr', {'end': 2, 'end_slope': 0}), ('SIr', [{'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
#
# # res = simplify_sigmoidal_intervals(('SI', {'start': 0.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
# # res = simplify_sigmoidal_intervals(('SI', {'start': -1.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
# res = simplify_sigmoidal_intervals(('SI', {'start': -1.5, 'end': 1.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]))
# print()
# print(res)