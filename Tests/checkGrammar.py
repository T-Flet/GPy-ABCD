from KernelExpansion.grammar import *


## unique

# b = [ChangeKE('CP', 'C', 'SE'), ChangeKE('CP', 'C', 'SE'), SumKE(['C', 'SE']), SumKE(['C', 'LIN'])]
# for e in unique(b):
#     print(e)


## Expansion

testExpr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
# print(testExpr._check_all_parents())
# a = flatten(times_base(testExpr.left.composite_terms[0]))
# a = expand_node(testExpr.left.composite_terms[0], production_rules_all)
# a = expand(testExpr, production_rules_all)
#
# for e in a: print(e)
# print(len(a))
# print(all([x._check_all_parents() for x in a]))


## Simplest expansion and standardised roots

# a = expand(SumKE(['WN'])._initialise(), production_rules_all)
# for e in a: print(e)
# print(len(a))
# print(all([x._check_all_parents() for x in a]))


## Uniqueness in expansions WITH unique REMOVED FROM THE FUNCTIONS THEMSELVES

# testExpr = ChangeKE('CP', ProductKE(['PER'], [SumKE(['WN', 'C', 'SE'])]), ChangeKE('CW', 'SE', ProductKE(['WN', 'LIN'])))._initialise()
# a = expand(testExpr, production_rules_all)
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


## Specific expansions

# a = expand(SumKE(['SE'])._initialise(), production_rules_all)
# a = expand(SumKE(['WN'])._initialise(), production_rules_all)
# a = expand(SumKE(['SE'])._initialise(), production_rules_start)
# a = expand(SumKE(['WN'])._initialise(), production_rules_start)
# a = standard_start_kernels

# for e in a: print(e)
# print(len(a))


## (PER + C) * (WN + C) -> (PER + C) * (C) and (PER + C) * (SE) and (PER + C) * (LIN) and just (PER + C)
# testExpr = ProductKE([], [SumKE(['PER', 'C']), SumKE(['WN', 'C'])])._initialise() # (PER + C) * (WN + C)
# a = [x.simplify() for x in expand(testExpr, [remove_some_term])]
# a = expand(testExpr, [remove_some_term])
# Adding the simplify() to "simplified = k_expr_root.simplify().extract_if_singleton()" in standardise_singleton_root solves this

# for e in a: print(e)
