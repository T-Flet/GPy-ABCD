from GPy_ABCD.KernelExpansion.kernelExpression import *
from GPy_ABCD.KernelExpansion.kernelExpressionOperations import *
import pytest


@pytest.mark.parametrize('a, b, op, res', [
    (SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C']), False, SumKE(['WN', 'PER', 'C'])),
    (ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE']), False, SumKE([], [ProductKE(['PER']), ProductKE(['LIN', 'SE'])])),
    (ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE'), False, SumKE([], [ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE')])),
    ('LIN', 'PER', False, SumKE(['LIN', 'PER'])),
    ('LIN', SumKE(['WN', 'C', 'C']), False, SumKE(['WN', 'LIN', 'C'])),
    ('LIN', ProductKE(['PER', 'SE', 'C']), False, SumKE(['LIN'], [ProductKE(['PER', 'SE'])])),  ## WITHOUT NESTED CASE HERE; SEE BELOW FOR WITH IT
    ('LIN', ChangeKE('CP', 'PER', 'C'), False, SumKE(['LIN'], [ChangeKE('CP', 'PER', 'C')])),
    (ProductKE(['PER', 'SE', 'C']), SumKE(['WN', 'C', 'C']), False, SumKE(['WN', 'C'], [ProductKE(['PER', 'SE'])])),
    (SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C'), False, SumKE(['WN', 'C'], [ChangeKE('CP', 'PER', 'C')])),
    (ProductKE(['PER', 'SE', 'C']), ChangeKE('CP', 'PER', 'C'), False, SumKE([], [ProductKE(['PER', 'SE']), ChangeKE('CP', 'PER', 'C')])),
    ('LIN', ProductKE(['PER', 'SE', 'C'], [ChangeKE('CP', 'C', 'LIN')]), True, [ # Only above case which is different for nondeterministic = True
        SumKE(['LIN'], [ProductKE(['PER', 'SE'], [ChangeKE('CP', 'C', 'LIN')])]),
        ProductKE(['SE'], [SumKE(['PER', 'LIN']), ChangeKE('CP', 'C', 'LIN')]),
        ProductKE(['PER'], [SumKE(['SE', 'LIN']), ChangeKE('CP', 'C', 'LIN')])
     ])
])
def test_add(a, b, op, res): assert add(a, b, op) == res


@pytest.mark.parametrize('a, b, op, res', [
    (SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C']), False, ProductKE([], [SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C'])])),
    (ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE']), False, ProductKE(['PER', 'LIN', 'C', 'SE'])),
    (ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE'), False, ProductKE([], [ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE')])),
    ('LIN', 'PER', False, ProductKE(['LIN', 'PER'])),
    ('LIN', SumKE(['WN', 'C', 'C']), False, ProductKE(['LIN'], [SumKE(['WN', 'C'])])), ## WITHOUT NESTED CASE HERE; SEE BELOW FOR WITH IT
    ('LIN', ProductKE(['PER', 'C']), False, ProductKE(['LIN', 'PER', 'C'])),
    ('LIN', ChangeKE('CP', 'PER', 'C'), False, ProductKE(['LIN'], [ChangeKE('CP', 'PER', 'C')])),
    (ProductKE(['PER', 'C']), SumKE(['WN', 'C', 'C']), False, ProductKE(['PER'], [SumKE(['WN', 'C'])])),
    (SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C'), False, ProductKE([], [SumKE(['WN', 'C']), ChangeKE('CP', 'PER', 'C')])),
    (ProductKE(['PER', 'C']), ChangeKE('CP', 'PER', 'C'), False, ProductKE(['PER'], [ChangeKE('CP', 'PER', 'C')])),
    ('LIN', SumKE(['WN', 'SE'], [ChangeKE('CP', 'C', 'LIN')]), True, [ # Only above case which is different for nondeterministic = True
        ProductKE(['LIN'], [SumKE(['WN', 'SE'], [ChangeKE('CP', 'C', 'LIN')])]),
        SumKE(['SE'], [ProductKE(['LIN', 'WN']), ChangeKE('CP', 'C', 'LIN')]),
        SumKE(['WN'], [ProductKE(['LIN', 'SE']), ChangeKE('CP', 'C', 'LIN')])
     ])
])
def test_multiply(a, b, op, res): assert multiply(a, b, op) == res


@pytest.mark.parametrize('kex, b, res', [
    (SumKE(['WN', 'PER', 'C']), 'SE', [SumKE(['SE', 'PER', 'C']), SumKE(['WN', 'SE', 'C']), SumKE(['WN', 'PER', 'SE'])]),
    (ProductKE(['LIN', 'PER', 'PER']), 'SE', [ProductKE(['SE', 'PER', 'PER']), ProductKE(['LIN', 'SE', 'PER'])]),
    (ChangeKE('CP', 'WN', SumKE(['PER', 'C'])), 'SE', [ChangeKE('CP', 'SE', SumKE(['PER', 'C']))]),
    (ChangeKE('CP', SumKE(['PER', 'C']), 'LIN'), 'SE', [ChangeKE('CP', SumKE(['PER', 'C']), 'SE')]),
    ('LIN', 'SE', ['SE'])
])
def test_swap_base(kex, b, res): assert swap_base(kex, b) == res


@pytest.mark.parametrize('kex, c, other, res', [
    (SumKE(['WN', 'PER', 'C']), 'CP', None, [ChangeKE('CP', SumKE(['WN', 'PER', 'C']), SumKE(['WN', 'PER', 'C']))]),
    (SumKE(['WN', 'PER', 'C']), 'CW', ProductKE(['PER', 'SE']), [ChangeKE('CW', SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE'])), ChangeKE('CW', ProductKE(['PER', 'SE']), SumKE(['WN', 'PER', 'C']))]),
    (SumKE(['WN', 'PER', 'C']), 'CP', 'WN', [ChangeKE('CP', SumKE(['WN', 'PER', 'C']), 'WN'), ChangeKE('CP', 'WN', SumKE(['WN', 'PER', 'C']))]),
])
def test_one_change(kex, c, other, res):
    if other is None: assert one_change(kex, c) == res
    else: assert one_change(kex, c, other) == res


@pytest.mark.parametrize('kex, other, res', [
    (SumKE(['WN', 'PER', 'C']), None, [ChangeKE('CP', SumKE(['WN', 'PER', 'C']), SumKE(['WN', 'PER', 'C'])), ChangeKE('CW', SumKE(['WN', 'PER', 'C']), SumKE(['WN', 'PER', 'C']))]),
    (SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE']), [ChangeKE('CP', SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE'])), ChangeKE('CP', ProductKE(['PER', 'SE']), SumKE(['WN', 'PER', 'C'])), ChangeKE('CW', SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE'])), ChangeKE('CW', ProductKE(['PER', 'SE']), SumKE(['WN', 'PER', 'C']))]),
    (SumKE(['WN', 'PER', 'C']), 'WN', [ChangeKE('CP', SumKE(['WN', 'PER', 'C']), 'WN'), ChangeKE('CP', 'WN', SumKE(['WN', 'PER', 'C'])), ChangeKE('CW', SumKE(['WN', 'PER', 'C']), 'WN'), ChangeKE('CW', 'WN', SumKE(['WN', 'PER', 'C']))]),
])
def test_both_changes(kex, other, res):
    if other is None: assert both_changes(kex) == res
    else: assert both_changes(kex, other) == res


@pytest.mark.parametrize('kex, res', [
    (SumKE(['WN', 'PER', 'C']), [SumKE(['PER', 'C']), SumKE(['WN', 'C']), SumKE(['WN', 'PER'])]),
    (ProductKE(['PER', 'SE', 'LIN']), [ProductKE(['SE', 'LIN']), ProductKE(['PER', 'LIN']), ProductKE(['PER', 'SE'])]),
    (ChangeKE('CP', SumKE(['WN', 'PER', 'C']), 'LIN'), [SumKE(['WN', 'PER', 'C']), SumKE(['LIN'])]),
    ('PER', [[]]) # This does not occur in expansions
])
def test_remove_a_term(kex, res): assert remove_a_term(kex) == res




@pytest.mark.by_inspection
class TestByInspection:
    @pytest.mark.parametrize('kex1, kex2', [
        (SumKE(['WN', 'C', 'C']), SumKE(['PER', 'C'])),
        (SumKE(['WN', 'C', 'C']), ProductKE(['PER', 'SE']))
    ])
    def test_sortOutTypePair(self, kex1, kex2):
        a = sortOutTypePair(kex1, kex2)
        print(a)
        print(a[SumKE])
