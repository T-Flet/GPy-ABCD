from Kernels.baseKernels import *


base_kerns = frozenset(['WN', 'C', 'LIN', 'SE', 'PER'])


# More efficient than eval(str) in the compositional evaluation (the other way if doing it one-off)
base_str_to_ker = {'PER': PER(), 'WN': WN(), 'SE': SE(), 'C': C(), 'LIN': LIN(), 'CP': CP, 'CW': CW}


base_k_param_names = {k: {'name': v.name, 'parameters': v.parameter_names()} for k, v in {B: base_str_to_ker[B] for B in base_kerns}.items()}


def remove_top_level_variance(ker):
    has_top_level_variance = 'variance' in ker.parameter_names()
    if has_top_level_variance: ker.unlink_parameter(ker.variance)
    return has_top_level_variance
# ker = LIN()
# # ker = CP(LIN(), WN())
# print(ker.parameter_names())
# print(remove_top_level_variance(ker))
# print(ker.parameter_names())
