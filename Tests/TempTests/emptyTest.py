from KernelExpansion.grammar import *
from KernelExpansion.kernelOperations import *

# a = ChangeKE('CW', SumKE(['LIN', 'PER']), SumKE(['LIN']))._initialise().traverse()
a = expand(ChangeKE('CW', 'LIN', 'LIN')._initialise(), production_rules_all)
for b in a: print(b)



# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
# ker = CP(LIN(), WN())
# # ker = testExpr.to_kernel()
# # print(ker.parameter_names())
# # print(ker.parameter_names_flat())
# # print(ker.parameters)
# # print(ker.param_array)
# print(get_param_dict(ker))
# ker.randomize()
# print(get_param_dict(ker))
#
#
# # print(attrgetter('linear_with_offset')(attrgetter('mul')(ker)))
# # print(ker['mul\.linear_with_offset'])
#
# # print(attrgetter('variance')(ker))
#
# print()
#
# # print(type(list(ker[r'.+variance'])[0]))
# # print(ker[r'.*variance'])







# ker = LIN()
# ker.unlink_parameter(ker.offset)
# # ker.offset = ker.variance.param_array
# ker.offset = ker.variance


# from modelSearch import *
# import numpy as np
#
# X = np.linspace(-10, 10, 101)[:, None]
#
# Y = (X - 5) / 2 + np.random.randn(101, 1) * 0.5 + 11
#
# from Util.util import doGPR
# doGPR(X, Y, ker, 3)
#
# print(ker.offset)
