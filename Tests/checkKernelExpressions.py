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


## Composite Singletons Simplifiaction

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

