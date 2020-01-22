from GPy_ABCD.KernelExpansion.kernelExpression import *
from GPy_ABCD.KernelExpansion.kernelExpressionOperations import *
from GPy_ABCD.KernelExpansion.kernelInterpretation import *
import pytest


@pytest.mark.parametrize('s_type, ss, res', [
    (S_overlap,
     [S_interval({'location': 0, 'slope': 0}),
     S_interval({'location': 1, 'slope': 1}),
     S_interval({'location': 2, 'slope': 2})],
     {'start': 2, 'start_slope': 2}),
    (Sr_overlap,
     [Sr_interval({'location': 0, 'slope': 0}),
      Sr_interval({'location': 1, 'slope': 1}),
      Sr_interval({'location': 2, 'slope': 2})],
     {'end': 0, 'end_slope': 0}),
    (SI_overlap,
     [SI_interval({'location': 0, 'slope': 0, 'width': 2}), # +-1
      SI_interval({'location': 1, 'slope': 1, 'width': 1}), # 0.5, 1.5
      SI_interval({'location': 0.25, 'slope': 2, 'width': 1})], # -0.25, 0.75
     {'start': 0.5, 'start_slope': 1, 'end': 0.75, 'end_slope': 2}),
    (SIr_overlap,
     [SIr_hole_interval({'location': 0, 'slope': 0, 'width': 2}), # +-1
      SIr_hole_interval({'location': 1, 'slope': 1, 'width': 1}), # 0.5, 1.5
      SIr_hole_interval({'location': 0.25, 'slope': 2, 'width': 1})], # -0.25, 0.75
     [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}])
])
def test_individual_sigmoid_overlaps(s_type, ss, res): assert s_type(ss) == res


@pytest.mark.parametrize('ss_tuple1, ss_tuple2, res', [
    (('S', {'start': 2, 'start_slope': 2}), ('SI', {'start': 1, 'end': 4, 'start_slope': 1, 'end_slope': 3}),
     ('SI', {'start': 2, 'end': 4, 'start_slope': 2, 'end_slope': 3})),
    (('S', {'start': 2, 'start_slope': 2}), ('Sr', {'end': 4, 'end_slope': 0}),
     ('SI', {'start': 2, 'start_slope': 2, 'end': 4, 'end_slope': 0})),
    (('S', {'start': 4, 'start_slope': 2}), ('Sr', {'end': 2, 'end_slope': 0}),
     ('SIr', [])),
    (('S', {'start': 2, 'start_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('S', {'start': 2, 'start_slope': 2})),
    (('S', {'start': 0.5, 'start_slope': 2}), ('SIr', [{'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('SIr', [{'start': 0.5, 'start_slope': 2, 'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}])),

    (('Sr', {'end': 1, 'end_slope': 0}), ('SI', {'start': 0.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}),
     ('SI', {'start': 0.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2})),
    (('Sr', {'end': 0, 'end_slope': 0}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('Sr', {'end': -1.0, 'end_slope': 0})),
    (('Sr', {'end': 2, 'end_slope': 0}), ('SIr', [{'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('SIr', [{'end': 1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1, 'end': 2, 'end_slope': 0}])),

    (('SI', {'start': 0.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('SIr', [])),
    (('SI', {'start': -1.5, 'end': 0.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('SI', {'start': -1.5, 'start_slope': 1, 'end': -1.0, 'end_slope': 0})),
    (('SI', {'start': -1.5, 'end': 1.75, 'start_slope': 1, 'end_slope': 2}), ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}]),
     ('SIr', [{'start': -1.5, 'start_slope': 1, 'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1, 'end': 1.75, 'end_slope': 2}]))
])
def test_simplify_sigmoidal_intervals_step(ss_tuple1, ss_tuple2, res): assert simplify_sigmoidal_intervals_step(ss_tuple1, ss_tuple2) == res


@pytest.mark.parametrize('interval_list, res', [
    ([('S', {'start': 1, 'start_slope': 2}),
     ('Sr', {'end': 2.5, 'end_slope': 0}),
     ('SI', {'start': 0.5, 'end': 3, 'start_slope': 1, 'end_slope': 3}),
     ('SIr', [{'end': 1.25, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}])],
     [{'start': 1, 'start_slope': 2, 'end': 1.25, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1, 'end': 2.5, 'end_slope': 0}]),
    ([('S', {'start': 1, 'start_slope': 2}),
     ('Sr', {'end': 2.5, 'end_slope': 0}),
     ('SI', {'start': 0.5, 'end': 3, 'start_slope': 1, 'end_slope': 3}),
     ('SIr', [{'end': -1.0, 'end_slope': 0}, {'start': 1.5, 'start_slope': 1}])],
     {'start': 1.5, 'start_slope': 1, 'end': 2.5, 'end_slope': 0})
])
def test_simplify_sigmoidal_intervals(interval_list, res): assert simplify_sigmoidal_intervals(dict(interval_list)) == res


@pytest.mark.by_inspection
class TestByInspection:

    @pytest.mark.parametrize('test_expr', [
        init_rand_params(ChangeKE('CW', 'LIN', 'PER')),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'SE'), SumKE(['C', 'WN'])])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'WN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])])),
        init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['SE', 'PER'])]))
    ])
    def test_full_interpretation(self, test_expr):
        print()
        res = test_expr.sum_of_prods_form()
        print(res)
        print(res.parameters)
        print()

        component_n = 1
        print(res.composite_terms[component_n].parameters)
        res = base_factors_interpretation(res.composite_terms[component_n].parameters)
        print(res)


    @pytest.mark.parametrize('test_expr', [
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'WN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])]))
    ])
    def test_first_term_interpretation(self, test_expr):
        res = test_expr.sum_of_prods_form()
        print(res)
        print(res.parameters)
        print(res.composite_terms[0].parameters)

        component_n = 2
        del res.composite_terms[component_n].parameters['ProductKE']
        ordered_ps = sorted(res.composite_terms[component_n].parameters.items(),
                            key=lambda bps: base_kern_interp_order[bps[0]])

        res = first_term_interpretation(ordered_ps[0])
        print(res)


    @pytest.mark.parametrize('test_expr', [
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'WN', 'PER'), SumKE(['C', 'PER'])])),
        init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])])),
        init_rand_params(ProductKE(['PER'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['SE', 'PER'])]))
    ])
    def test_postmodifier_term_interpretation(self, test_expr):
        res = test_expr.sum_of_prods_form()
        print(res)
        # print(res.parameters)
        # print(res.composite_terms[0].parameters)
        print()

        component_n = 3
        del res.composite_terms[component_n].parameters['ProductKE']
        ordered_ps = sorted(res.composite_terms[component_n].parameters.items(),
                            key=lambda bps: base_kern_interp_order[bps[0]])
        print(ordered_ps)

        res = postmodifier_interpretation(ordered_ps[1])
        print(res)
        print('Res: ^.^')
