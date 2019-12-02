from KernelExpansion.kernelExpression import *
from KernelExpansion.kernelExpressionOperations import *


## Base Terms Simplification

# aTest = SumKE(['WN', 'WN', 'C', 'C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)
#
# aTest = ProductKE(['C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)
# aTest.new_base('WN')
# print(aTest)


## Type and Printing

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))
# testKern = testExpr.to_kernel()

# print(testExpr)
# print(testKern)
#
# print(isinstance(testExpr, KernelExpression))
# from GPy.kern.src.kern import Kern
# print(isinstance(testKern, Kern))


## Root and Parents

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))
# testExpr.set_root(testExpr)
# for kex in testExpr.traverse(): print(kex.root)

# testExpr._set_all_parents()
# testExpr.left.parent = testExpr.right.parent = testExpr
# testExpr.left.composite_terms[0].parent = testExpr.left
# testExpr.right.right.parent = testExpr.right
# print(testExpr._check_all_parents())

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))._initialise()
# for kex in testExpr.traverse(): print(kex.root)
# print(testExpr._check_all_parents())


## Composite Singletons Simplification

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))
# testExpr.set_root(testExpr)._set_all_parents() # I.e. an _initialise without .simplify()
#
# print(testExpr.right)
# print(type(testExpr.right.right))
# testExpr.right.simplify()
# print(type(testExpr.right.right))
# print(testExpr.right)
#
# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'C'])))._initialise()
# print(type(testExpr.right.right))
# print(testExpr)
#
# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['WN', 'C'])]))._initialise()
# print(type(testExpr.right)) # Two nested singleton extractions occurred
# print(testExpr)
#
# testExpr = SumKE([], [ProductKE([], [SumKE([], [ProductKE(['LIN'],[])])])])._initialise()
# print(testExpr)
# print(type(testExpr))
# print(testExpr.composite_terms)
#
# testExpr = SumKE([], [ProductKE([], [SumKE([], [ProductKE(['LIN', 'SE'],[])])])])._initialise()
# print(testExpr)
# print(type(testExpr))
# print(testExpr.composite_terms)


## Homogeneous Composites Simplification

# testExpr = SumKE(['PER', 'SE'], [SumKE(['WN', 'C', 'C'], [SumKE(['SE'], [])]), ProductKE(['LIN', 'WN'], [])])._initialise()
# print(testExpr)
# print(type(testExpr))
# print(testExpr.composite_terms)


## Edge case simplifications
# Lists of unhashables
# a = ProductKE([], [SumKE(['PER', 'C']), SumKE(['WN', 'C'])]) # (PER + C) * (WN + C)
# b = ProductKE([], [SumKE(['WN', 'C']), SumKE(['PER', 'C'])]) # (WN + C) * (PER + C)
# print(a == b)


## Traverse and Reduce

# testTraversed = testExpr.traverse()
# print([str(x) for x in testTraversed])

# def testFunc(node, acc):
#     node.set_root('HI')
#     if isinstance(node, SumOrProductKE): # Or split Sum and Product cases further
#         node.new_base('LIN')
#         acc += node.base_terms.elements()
#     else: # elif isinstance(node, ChangeKE):
#         if isinstance(node.left, str): acc += [node.left]
#         if isinstance(node.right, str): acc += [node.right]
#     return acc
#
# a = testExpr.reduce(testFunc, [])
# for kex in testExpr.traverse(): print(kex.root)
# print(testExpr)
# print(a)


## Root a and Deepcopy Tests (Root part required, otherwise root is None)
# from copy import deepcopy
#
# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
# print(testExpr.root)
# testExpr.set_root(testExpr)
# print(testExpr.left.root)

# print(testExpr.root is deepcopy(testExpr).root)
# print(testExpr.left.root is deepcopy(testExpr.left).root)
# print(deepcopy(testExpr).root is deepcopy(testExpr).root)
# dcTE = deepcopy(testExpr)
# print(dcTE.root is dcTE.root)
# print(dcTE.root is dcTE.root.root)
# dcTE.left.set_root(dcTE.right)
# dcTE.right.set_root(dcTE.left)
# dcdcTE = deepcopy(dcTE)
# print(dcdcTE.left.root is dcdcTE.right)
#
# testTraversed = testExpr.traverse()
# dcTestTraversed = deepcopy(testTraversed)
# print(testTraversed[1].root is dcTestTraversed[1].root)
# print(testTraversed[0].root is dcTestTraversed[1].root)


## __eq__ Tests
# from copy import deepcopy
#
# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
# print(testExpr is deepcopy(testExpr))
# print(testExpr == deepcopy(testExpr))
# print(deepcopy(testExpr) is deepcopy(testExpr))
# print(deepcopy(testExpr) == deepcopy(testExpr))


## Overloaded + * Tests NOT CURRENTLY IMPLEMENTED
# print(SumKE(['WN', 'C', 'C']) + SumKE(['WN', 'PER', 'C']))
# print(SumKE(['WN', 'C', 'C']) * SumKE(['WN', 'PER', 'C']))


## to_kernel_by_parts

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
# # testExpr = ProductKE(['WN'], [SumKE(['PER', 'C'])])._initialise()
# ker_by_parts = testExpr.to_kernel()
# ker_by_eval = testExpr.to_kernel_unrefined()
# print(ker_by_parts)
# print(len(ker_by_parts.parameter_names()))
# print(ker_by_eval)
# print(len(ker_by_eval.parameter_names()))


## match_up_fit_parameters

# # testExpr = SumKE(['WN', 'PER', 'C'])._initialise()
# # testExpr = SumKE(['WN', 'PER', 'PER'])._initialise()
# # testExpr = SumKE([], [ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER'])])._initialise()
# # testExpr = SumKE(['WN', 'PER', 'C'], [ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER']), ProductKE(['SE', 'PER'])])._initialise()
# # testExpr = ProductKE(['SE', 'PER'])._initialise()
# # testExpr = ProductKE(['SE', 'PER', 'PER'])._initialise()
# # testExpr = ProductKE([], [SumKE(['SE', 'PER']), SumKE(['SE', 'PER']), SumKE(['SE', 'PER'])])._initialise()
# # testExpr = ProductKE(['SE', 'PER'], [SumKE(['SE', 'PER']), SumKE(['SE', 'PER']), SumKE(['SE', 'PER'])])._initialise()
# # testExpr = ChangeKE('CP', 'PER', SumKE(['C', 'PER']))._initialise()
# # testExpr = ChangeKE('CW', 'PER', SumKE(['C', 'PER']))._initialise()
# # testExpr = ChangeKE('CW', 'LIN', 'LIN')._initialise()
# testExpr = ChangeKE('CW', SumKE(['C', 'PER']), SumKE(['C', 'PER']))._initialise()
#
# print(testExpr)
# ker = testExpr.to_kernel()
# ker.randomize()
# param_dict = get_param_dict(ker)
# print(param_dict)
# res = testExpr.match_up_fit_parameters(param_dict, '')
# print(res.parameters)
# # print(res.composite_terms[2].parameters)
# print(res.right.parameters)


## sum_of_prods_form

# testExpr = init_rand_params(ChangeKE('CP', 'LIN', 'PER'))
# testExpr = init_rand_params(ChangeKE('CW', 'LIN', 'PER'))
# testExpr = init_rand_params(SumKE(['LIN', 'PER']))
# testExpr = init_rand_params(ProductKE(['LIN', 'PER']))
testExpr = init_rand_params(ProductKE(['SE'], [SumKE(['LIN', 'PER'])]))



# testExpr = init_rand_params(ProductKE(['PER'])).new_bases_with_parameters([('SE', {'variance': x, 'lengthscale': x}) for x in (2,3,5)])
# testExpr = init_rand_params(ProductKE(['PER'])).new_bases_with_parameters(('WN', {'variance': 2}))
# testExpr = init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('C', {'variance': 2}))
# testExpr = init_rand_params(ProductKE(['SE'])).new_bases_with_parameters(('C', {'variance': 2}))
# testExpr = init_rand_params(ProductKE(['WN'])).new_bases_with_parameters(('C', {'variance': 2}))
# testExpr = init_rand_params(ProductKE(['C'])).new_bases_with_parameters(('WN', {'variance': 2}))



res = testExpr.sum_of_prods_form()
print(res)
print(res.parameters)
print(res.composite_terms[0].parameters)
# print(res.right.parameters)




#### kernelExpressionOperators

## sortOutTypePair
# a = sortOutTypePair(SumKE(['WN', 'C', 'C']), SumKE(['PER', 'C']))
# print(a)
# print(a[SumKE])

## add
# print(add(SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C']), False))
# print(add(ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE']), False))
# print(add(ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE'), False))
# print(add('LIN', 'PER', False))
# print(add('LIN', SumKE(['WN', 'C', 'C']), False))
# print(add('LIN', ProductKE(['PER', 'SE', 'C']), False)) ## WITHOUT NESTED CASE HERE; SEE BELOW FOR WITH IT
# print(add('LIN', ChangeKE('CP', 'PER', 'C'), False))
# print(add(ProductKE(['PER', 'SE', 'C']), SumKE(['WN', 'C', 'C']), False))
# print(add(SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C'), False))
# print(add(ProductKE(['PER', 'SE', 'C']), ChangeKE('CP', 'PER', 'C'), False))

# Only above case which is different for nondeterministic = True
# for r in add('LIN', ProductKE(['PER', 'SE', 'C'], [ChangeKE('CP', 'C', 'LIN')])): print(r)


## multiply
# print(multiply(SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C']), False))
# print(multiply(ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE']), False))
# print(multiply(ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE'), False))
# print(multiply('LIN', 'PER', False))
# print(multiply('LIN', SumKE(['WN', 'C', 'C']), False)) ## WITHOUT NESTED CASE HERE; SEE BELOW FOR WITH IT
# print(multiply('LIN', ProductKE(['PER', 'C']), False))
# print(multiply('LIN', ChangeKE('CP', 'PER', 'C'), False))
# print(multiply(ProductKE(['PER', 'C']), SumKE(['WN', 'C', 'C']), False))
# print(multiply(SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C'), False))
# print(multiply(ProductKE(['PER', 'C']), ChangeKE('CP', 'PER', 'C'), False))

# Only above case which is different for nondeterministic = True
# for r in multiply('LIN', SumKE(['WN', 'SE'], [ChangeKE('CP', 'C', 'LIN')])): print(r)


## swap_base
# for e in swap_base(SumKE(['WN', 'PER', 'C']), 'SE'): print(e)
# for e in swap_base(ProductKE(['LIN', 'PER', 'PER']), 'SE'): print(e)
# for e in swap_base(ChangeKE('CP', 'WN', SumKE(['PER', 'C'])), 'SE'): print(e)
# for e in swap_base(ChangeKE('CP', SumKE(['PER', 'C']), 'LIN'), 'SE'): print(e)
# for e in swap_base('LIN', 'SE'): print(e)


## one_change and both_changes
# for e in one_change(SumKE(['WN', 'PER', 'C']), 'CP'): print(e)
# for e in one_change(SumKE(['WN', 'PER', 'C']), 'CW', ProductKE(['PER', 'SE'])): print(e)
# for e in one_change(SumKE(['WN', 'PER', 'C']), 'CP', 'WN'): print(e)

# for e in both_changes(SumKE(['WN', 'PER', 'C'])): print(e)
# for e in both_changes(SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE'])): print(e)
# for e in both_changes(SumKE(['WN', 'PER', 'C']), 'WN'): print(e)


## replace_node
# for e in replace_node(SumKE(['WN', 'PER', 'C']), ProductKE(['PER', 'SE'])): print(e)
# for e in replace_node(SumKE(['WN', 'PER', 'C']), 'WN'): print(e)


## remove_a_term
# for e in remove_a_term(SumKE(['WN', 'PER', 'C'])): print(e)
# for e in remove_a_term(ProductKE(['PER', 'SE', 'LIN'])): print(e)
# for e in remove_a_term(ChangeKE('CP', SumKE(['WN', 'PER', 'C']), 'LIN')): print(str(e) + str(type(e)))
# for e in remove_a_term('PER'): print(e) # This does not occur in expansions


## add_sum_of_prods_terms
# a = SumKE([], [ProductKE(['SE', 'PER'])])
# b = ProductKE(['S'])
# print(add_sum_of_prods_terms(a,a))
# print(add_sum_of_prods_terms(a,b))
# print(add_sum_of_prods_terms(b,a))
# print(add_sum_of_prods_terms(b,b))

