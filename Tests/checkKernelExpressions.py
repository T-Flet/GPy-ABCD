from kernelExpression import *
from kernelExpressionOperations import *


### Simplification Tests

## SumKE
# aTest = SumKE(['WN', 'WN', 'C', 'C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)

## ProductKE
# aTest = ProductKE(['C', 'LIN', 'LIN', 'SE', 'SE', 'PER', 'PER'])
# print(aTest)
# aTest.new_base('WN')
# print(aTest)



### General Tests

# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE(['SE', 'WN']))
# testKern = testExpr.to_kernel()
#
# print(testExpr)
# print(testKern)
#
# print(isinstance(testExpr, KernelExpression))
# from GPy.kern.src.kern import Kern
# print(isinstance(testKern, Kern))
#
# testTraversed = testExpr.traverse()
# print([str(x) for x in testTraversed])
#
# print(testExpr.root)
# testExpr.set_root(testExpr)
# print(testExpr.left.root)


## Copy and Deepcopy Tests
# from copy import copy, deepcopy
# print(testExpr.left.root == deepcopy(testExpr.left).root)
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
