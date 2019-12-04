from KernelExpansion.kernelExpression import *
from KernelExpansion.kernelExpressionOperations import *


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
