from GPy_ABCD.KernelExpansion.grammar import *
import pytest


@pytest.mark.parametrize('kex, kex_str', [
    (ChangeKE('CP', 'C', 'SE'), 'CP(C, SE)'),
    (SumKE(['C', 'SE']), 'SE + C'),
    (ProductKE(['PER', 'LIN']), 'PER * LIN')
])
def test_to_str(kex, kex_str): assert str(kex) == kex_str


def test_unique():
    non_unique = [ChangeKE('CP', 'C', 'SE'), ChangeKE('CP', 'C', 'SE'), SumKE(['C', 'SE']), SumKE(['C', 'LIN'])]
    assert unique(non_unique) == non_unique[1:]


def test_unique_deeper(): # Uniqueness in expansions WITH unique REMOVED FROM THE FUNCTIONS THEMSELVES
    test_expr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]),
                         ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
    a = expand(test_expr, production_rules_all)
    astr = [str(x) for x in a]
    aUnique = unique(a)
    aUniqueStr = [str(x) for x in aUnique]
    assert len(astr) == len(aUniqueStr)

    aDiff = deepcopy(a)
    for x in aUnique: aDiff.remove(x)
    assert not aDiff

    assert set(astr) == set(aUniqueStr)




@pytest.mark.by_inspection
class TestByInspection:
    # Expansion checks are not tests because the production rules might change

    def test_simplest_expansion(self):
        print()
        es = expand(SumKE(['WN'])._initialise(), production_rules_all)
        for e in es: print(e)
        print(len(es))
        print(all([x._check_all_parents() for x in es]))


    @pytest.mark.parametrize('es', [
        expand(SumKE(['SE'])._initialise(), production_rules_all),
        expand(SumKE(['WN'])._initialise(), production_rules_all),
        # expand(SumKE(['SE'])._initialise(), production_rules_start),
        # expand(SumKE(['WN'])._initialise(), production_rules_start),
        standard_start_kernels
    ])
    def test_start_expansions(self, es):
        print()
        for e in es: print(e)
        print(len(es))


    test_expr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
    @pytest.mark.parametrize('es', [
        flatten(times_base(test_expr.left.composite_terms[0])),
        expand_node(test_expr.left.composite_terms[0], production_rules_all),
        expand(test_expr, production_rules_all)
    ])
    def test_long_expansions(self, es):
        assert self.test_expr._check_all_parents()
        print()
        for e in es: print(e)
        print(len(es))
        assert all([x._check_all_parents() for x in es])


    # (PER + C) * (WN + C) -> (PER + C) * (C) and (PER + C) * (SE) and (PER + C) * (LIN) and just (PER + C)
    test_expr = ProductKE([], [SumKE(['PER', 'C']), SumKE(['WN', 'C'])])._initialise()  # (PER + C) * (WN + C)
    @pytest.mark.parametrize('es', [
        [x.simplify() for x in expand(test_expr, [remove_some_term])],
        expand(test_expr, [remove_some_term])
        # Adding the simplify() to "simplified = k_expr_root.simplify().extract_if_singleton()" in standardise_singleton_root solves this
    ])
    def test_specific_expansions(self, es):
        print()
        for e in es: print(e)
        print(len(es))

