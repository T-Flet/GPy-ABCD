from kernelExpression import *
from kernelExpressionOperations import *


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
# print(testExpr.root)
# testExpr.set_root(testExpr)
# print(testExpr.left.root)
#
# print(testExpr.root == deepcopy(testExpr).root)
# print(testExpr.left.root == deepcopy(testExpr.left).root)
# print(deepcopy(testExpr).root == deepcopy(testExpr).root)
# dcTE = deepcopy(testExpr)
# print(dcTE.root == dcTE.root)
# print(dcTE.root == dcTE.root.root)
# dcTE.left.set_root(dcTE.right)
# dcTE.right.set_root(dcTE.left)
# dcdcTE = deepcopy(dcTE)
# print(dcdcTE.left.root == dcdcTE.right)
#
# dcTestTraversed = deepcopy(testTraversed)
# print(testTraversed[1].root == dcTestTraversed[1].root)
# print(testTraversed[0].root == dcTestTraversed[1].root)



## Overloading Tests NOT CURRENTLY IMPLEMENTED
# print(SumKE(['WN', 'C', 'C']) + SumKE(['WN', 'PER', 'C']))
# print(SumKE(['WN', 'C', 'C']) * SumKE(['WN', 'PER', 'C']))




#### kernelExpressionOperators

## sortOutTypePair
# a = sortOutTypePair(SumKE(['WN', 'C', 'C']), SumKE(['PER', 'C']))
# print(a)
# print(a[SumKE])

## add
# print(add(SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C'])))
# print(add(ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE'])))
# print(add(ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE')))
# print(add('LIN', 'PER'))
# print(add('LIN', SumKE(['WN', 'C', 'C'])))
# print(add('LIN', ProductKE(['PER', 'SE', 'C'])))
# print(add('LIN', ChangeKE('CP', 'PER', 'C')))
# print(add(ProductKE(['PER', 'SE', 'C']), SumKE(['WN', 'C', 'C'])))
# print(add(SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C')))
# print(add(ProductKE(['PER', 'SE', 'C']), ChangeKE('CP', 'PER', 'C')))

## multiply
# print(multiply(SumKE(['WN', 'C', 'C']), SumKE(['WN', 'PER', 'C'])))
# print(multiply(ProductKE(['PER', 'C']), ProductKE(['LIN', 'C', 'SE'])))
# print(multiply(ChangeKE('CP', 'PER', 'C'), ChangeKE('CW', 'LIN', 'SE')))
# print(multiply('LIN', 'PER'))
# print(multiply('LIN', SumKE(['WN', 'C', 'C'])))
# print(multiply('LIN', ProductKE(['PER', 'C'])))
# print(multiply('LIN', ChangeKE('CP', 'PER', 'C')))
# print(multiply(ProductKE(['PER', 'C']), SumKE(['WN', 'C', 'C'])))
# print(multiply(SumKE(['WN', 'C', 'C']), ChangeKE('CP', 'PER', 'C')))
# print(multiply(ProductKE(['PER', 'C']), ChangeKE('CP', 'PER', 'C')))
