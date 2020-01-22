from GPy_ABCD.KernelExpansion.kernelExpressionOperations import *
from GPy_ABCD.Util.genericUtil import diff
import pytest


# No point in making this a fixture since it will be modified each time and is also needed in parametrisations
base_expr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))


@pytest.mark.parametrize('kex, res', [
    # Base terms
    (SumKE(['WN', 'WN', 'C', 'C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER']), SumKE(['WN', 'C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])),
    (ProductKE(['C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER']), ProductKE(['LIN', 'LIN', 'SE', 'PER', 'PER'])),
    (ProductKE(['C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER', 'WN']), ProductKE(['LIN', 'LIN', 'WN'])),
    # Nested Singleton Extractions
        # Base term singletons
    (ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['WN', 'C'])]))._initialise(), ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C'])]), 'WN')._initialise()),
    (SumKE([], [ProductKE([], [SumKE([], [ProductKE(['LIN'],[])])])])._initialise(), SumKE(['LIN'])),
        # Composite term singletons
    (ProductKE([], [SumKE([], [ProductKE(['LIN', 'SE'])])])._initialise(), ProductKE(['LIN', 'SE'])._initialise()),
    (SumKE([], [ProductKE([], [SumKE([], [ProductKE(['LIN', 'SE'])])])])._initialise(), SumKE([], [ProductKE(['LIN', 'SE'],[])])._initialise()), # This one cannot go further by itself; standardise_singleton_root in grammar handles it
        # Homogeneous Composites
    (SumKE(['PER', 'SE'], [SumKE(['WN', 'C', 'C'], [SumKE(['SE'], [])]), ProductKE(['LIN', 'WN'], [])])._initialise(), SumKE(['PER', 'SE', 'SE', 'WN', 'C'], [ProductKE(['LIN', 'WN'])]))
])
def test_simplification(kex, res): assert kex == res


@pytest.mark.parametrize('kex', [
    SumKE(['LIN']),
    SumKE(['LIN', 'LIN']),
    SumKE(['LIN'], [ProductKE(['SE'])]),
    SumKE(['LIN'], [ChangeKE('CP', 'PER', ProductKE(['WN']))]),
    base_expr
])
def test_repr(kex): # Whether the repr can be parsed to get an equal object
    assert eval(kex.__repr__()) == kex


def test_root_parents__initialise():
    test_expr = deepcopy(base_expr)
    test_expr.set_root(test_expr)
    assert all([kex.root == test_expr.root for kex in test_expr.traverse()])

    test_expr.left.parent = test_expr.right.parent = test_expr
    test_expr.left.composite_terms[0].parent = test_expr.left
    assert test_expr._check_all_parents()
    test_expr._set_all_parents()
    assert test_expr._check_all_parents()

    # The above is technically and ._initialise without .simplify (although base_expr already autosimplified)
    test_expr = deepcopy(base_expr)._initialise()
    assert all([kex.root == test_expr.root for kex in test_expr.traverse()])
    assert test_expr._check_all_parents()


def test_reduce():
    test_expr = deepcopy(base_expr)
    def testFunc(node, acc): # Sets arbitrary root, adds LIN base term and returns all base terms
        node.set_root('HI')
        if isinstance(node, SumOrProductKE):  # Or split Sum and Product cases further
            node.new_base('LIN')
            acc += node.base_terms.elements()
        else:  # elif isinstance(node, ChangeKE):
            if isinstance(node.left, str): acc += [node.left]
            if isinstance(node.right, str): acc += [node.right]
        return acc

    res = test_expr.reduce(testFunc, [])
    assert res == ['PER', 'LIN', 'WN', 'C', 'LIN', 'SE', 'WN']
    assert [kex.root == 'HI' for kex in test_expr.traverse()]
    assert test_expr == ChangeKE('CP', ProductKE(['PER', 'LIN'], [SumKE(['WN', 'C', 'LIN'])]), ChangeKE('CW', 'SE', 'WN'))


def test__eq():
    test_expr = deepcopy(base_expr)

    assert test_expr is not deepcopy(test_expr)
    assert test_expr == deepcopy(test_expr)

    assert deepcopy(test_expr) is not deepcopy(test_expr)
    assert deepcopy(test_expr) == deepcopy(test_expr)

    # Lists of unhashables
    a = ProductKE([], [SumKE(['PER', 'C']), SumKE(['WN', 'C'])])  # (PER + C) * (WN + C)
    b = ProductKE([], [SumKE(['WN', 'C']), SumKE(['PER', 'C'])])  # (WN + C) * (PER + C)
    assert a == b

    ## Overloaded + * Tests NOT CURRENTLY IMPLEMENTED
    # print(SumKE(['WN', 'C', 'C']) + SumKE(['WN', 'PER', 'C']))
    # print(SumKE(['WN', 'C', 'C']) * SumKE(['WN', 'PER', 'C']))


def test_to_kernel():
    test_expr = deepcopy(base_expr)._initialise()
    ker_by_parts = test_expr.to_kernel()
    ker_by_eval = test_expr.to_kernel_unrefined()
    assert ker_by_parts.parameter_names() == ker_by_eval.parameter_names()

    # Case showing the removal of the extra variance of base terms' multiplication
    test_expr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
    ker_by_parts = test_expr.to_kernel()
    ker_by_eval = test_expr.to_kernel_unrefined()
    assert diff(ker_by_eval.parameter_names(), ker_by_parts.parameter_names()) == ['mul_1.linear_with_offset.variance']


def test_deepcopy_root(): # Not really a test of this library's functionality; really just of deepcopy to make sure
    test_expr = deepcopy(base_expr)._initialise()

    assert test_expr.root is not deepcopy(test_expr).root
    assert test_expr.left.root is not deepcopy(test_expr.left).root
    assert deepcopy(test_expr).root is not deepcopy(test_expr).root

    dcTE = deepcopy(test_expr)
    assert dcTE.root is dcTE.root
    assert dcTE.root is dcTE.root.root

    dcTE.left.set_root(dcTE.right)
    dcTE.right.set_root(dcTE.left)
    dcdcTE = deepcopy(dcTE)
    assert dcdcTE.left.root is dcdcTE.right

    testTraversed = test_expr.traverse()
    dcTestTraversed = deepcopy(testTraversed)
    assert testTraversed[1].root is not dcTestTraversed[1].root
    assert testTraversed[0].root is not dcTestTraversed[1].root



@pytest.mark.by_inspection
class TestByInspection:

    def test_type_and_str(self):
        test_expr = deepcopy(base_expr)
        testKern = test_expr.to_kernel()

        assert isinstance(test_expr, KernelExpression)
        from GPy.kern.src.kern import Kern
        assert isinstance(testKern, Kern)

        print()
        print(test_expr)
        print(testKern)


    def test_traverse(self):
        test_expr = deepcopy(base_expr)
        print()
        print(test_expr)
        print([str(x) for x in test_expr.traverse()])

    # Thoroughly check that the parameters end up where they should based on their long names
    @pytest.mark.parametrize('kex', [
        SumKE(['WN', 'PER', 'C']),
        SumKE(['WN', 'PER', 'PER']),
        SumKE([], [ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER'])]),
        SumKE(['WN', 'PER', 'C'], [ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER'])]),
        ProductKE(['SE', 'PER']),
        ProductKE(['SE', 'PER', 'PER']),
        ProductKE([], [SumKE(['SE', 'PER']), SumKE(['SE', 'PER']), SumKE(['SE', 'PER'])]),
        ProductKE(['SE', 'PER'], [SumKE(['SE', 'PER']), SumKE(['SE', 'PER']), SumKE(['SE', 'PER'])]),
        ChangeKE('CP', 'PER', SumKE(['C', 'PER'])),
        ChangeKE('CW', 'PER', SumKE(['C', 'PER'])),
        ChangeKE('CW', 'LIN', 'LIN'),
        ChangeKE('CW', SumKE(['C', 'PER']), SumKE(['C', 'PER']))._initialise()
    ])
    def test_absorb_fit_parameters(self, kex):
        kex._initialise()
        print('\n')
        print(kex)
        ker = kex.to_kernel()
        ker.randomize()
        param_dict = get_param_dict(ker)
        print(param_dict)
        res = kex.match_up_fit_parameters(param_dict, '')
        for r in res.traverse(): print(r.parameters)


    # Check that there is only one variance (equal to the product of all)
    def test_multiply_pure_prods_with_params(self):
        print()
        args = [init_rand_params(ProductKE(bts)) for bts in (['LIN', 'SE'], ['PER'], ['SE'])]
        res = ProductKE.multiply_pure_prods_with_params(args[0], args[1:])
        print()
        print(res)
        print(res.parameters)


    # Thoroughly check that:
    #   - the sum-of-product form is correct
    #   - the parameters look appropriate for their expression
    @pytest.mark.parametrize('kex', [
        init_rand_params(ChangeKE('CP', 'LIN', 'PER')),
        init_rand_params(ChangeKE('CW', 'LIN', 'PER')),

        init_rand_params(SumKE(['LIN', 'PER'])),
        init_rand_params(SumKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), ProductKE(['LIN', 'PER'])])),

        init_rand_params(ProductKE(['PER'])),
        init_rand_params(ProductKE(['SE'], [SumKE(['C', 'LIN'])])),
        init_rand_params(ProductKE(['LIN', 'PER'])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])])),

        init_rand_params(ProductKE(['PER'])).new_bases_with_parameters([('SE', {'variance': x, 'lengthscale': x}) for x in (2,3,5)]),
        init_rand_params(ProductKE(['PER'])).new_bases_with_parameters(('WN', {'variance': 2})),
        init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['SE'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['WN'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('WN', {'variance': 2}))
    ])
    def test_sum_of_prods_form(self, kex):
        print('\n')
        print(kex)
        res = kex.sum_of_prods_form()
        print(res)
        print(res.composite_terms)
        print(res.parameters)
        for pt in res.composite_terms: print(pt.parameters)


    # Thoroughly check that the four combinations produce correct outputs
    @pytest.mark.parametrize('kex1', [
        SumKE([], [ProductKE(['SE', 'PER'])]),
        SumKE(['LIN', 'LIN'], [ProductKE(['SE', 'PER'])])
    ])
    def test_add_sum_of_prods_terms(self, kex1):
        print()
        kex1 = init_rand_params(kex1)
        kex2 = ProductKE(['S'])
        print(kex2)
        print()
        print(ChangeKE.add_sum_of_prods_terms(kex1,kex1))
        print(ChangeKE.add_sum_of_prods_terms(kex1,kex2))
        print(ChangeKE.add_sum_of_prods_terms(kex2,kex1))
        print(ChangeKE.add_sum_of_prods_terms(kex2,kex2))


    # Thoroughly check that the interpretations makes sense and has the correct parameters (approximated)
    @pytest.mark.parametrize('kex', [
        init_rand_params(ChangeKE('CP', 'LIN', 'PER')),
        init_rand_params(ChangeKE('CW', 'LIN', 'PER')),

        init_rand_params(SumKE(['LIN', 'PER'])),
        init_rand_params(SumKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), ProductKE(['LIN', 'PER'])])),

        init_rand_params(ProductKE(['PER'])).new_bases_with_parameters([('SE', {'variance': x, 'lengthscale': x}) for x in (2,3,5)]),
        init_rand_params(ProductKE(['PER'])).new_bases_with_parameters(('WN', {'variance': 2})),
        init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['SE'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['WN'])).new_bases_with_parameters(('C', {'variance': 2})),
        init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('WN', {'variance': 2})),

        init_rand_params(ProductKE(['PER'])),
        init_rand_params(ProductKE(['LIN', 'PER'])),
        init_rand_params(ProductKE(['SE'], [ChangeKE('CP', 'LIN', 'PER'), SumKE(['LIN', 'PER'])]))
    ])
    def test_get_interpretation(self, kex):
        print()
        print(kex)
        for k in kex.traverse(): print(k.parameters)
        print(kex.get_interpretation())
