from KernelExpansion.grammar import *


## unique

# b = [ChangeKE('CP', 'C', 'SE'), ChangeKE('CP', 'C', 'SE'), SumKE(['C', 'SE']), SumKE(['C', 'LIN'])]
# for e in unique(b):
#     print(e)


## Expansion

testExpr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
# a = flatten(times_base(testExpr.left.composite_terms[0]))
# a = expand_node(testExpr.left.composite_terms[0], production_rules.values())
# a = expand_node(testExpr, production_rules.values())
# a = expand(testExpr, production_rules.values())
#
# for e in a: print(e)
# print(len(a))
# print(all([x._check_all_parents() for x in a]))


## Simplest expansion and standardised roots

# a = expand(SumKE(['WN'])._initialise(), production_rules.values())
# for e in a: print(e)
# print(len(a))
# print(all([x._check_all_parents() for x in a]))


## Uniqueness in expansions WITH unique REMOVED FROM THE FUNCTIONS THEMSELVES

# testExpr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
# a = expand(testExpr, production_rules.values())
# astr = [str(x) for x in a]
# aUnique = unique(a)
# aUniqueStr = [str(x) for x in aUnique]
# print(len(astr))
# print(len(aUniqueStr))
#
# print()
# aDiff = deepcopy(a)
# for x in aUnique: aDiff.remove(x)
# for e in aDiff: print(e)
# print(len(aDiff))
# print()
#
# print(len(set(astr)))
# print(len(set(aUniqueStr)))
# print(set(astr) - set(aUniqueStr))
