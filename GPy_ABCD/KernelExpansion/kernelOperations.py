from GPy_ABCD.Kernels import *


base_kerns = frozenset(['WN', 'C', 'LIN', 'SE', 'PER'])
base_sigmoids = frozenset(['S', 'Sr', 'SI', 'SIr'])

# More efficient than eval(str) in the compositional evaluation (the other way if doing it one-off)
base_str_to_ker_func = {'PER': PER, 'WN': WN, 'SE': SE, 'C': C, 'LIN': LIN, 'CP': CP, 'CW': CW}
def base_str_to_ker(base_ker_str): return base_str_to_ker_func[base_ker_str]()


base_k_param_names = {k: {'name': v.name, 'parameters': v.parameter_names()} for k, v in {B: base_str_to_ker(B) for B in base_kerns}.items()}
base_k_param_names['CP'] = {'name': 'change_point', 'parameters': ['location', 'slope']}
base_k_param_names['CW'] = {'name': 'change_window', 'parameters': ['location', 'slope', 'width']} # Nothing, 'stop_location' or 'width' depending on the used class


change_k_sigmoid_names = {'CP': {'left': 'Sr', 'right': 'S'}, 'CW': {'left': 'SIr', 'right': 'SI'}}


def get_param_dict(fit_ker):
    full_names = fit_ker.parameter_names_flat()
    values = fit_ker.param_array
    return dict(zip(full_names, values))
# testExpr = ChangeKE('CP', ProductKE(['PER', 'C'], [SumKE(['WN', 'C', 'C'])]), SumKE([], [ProductKE(['SE', 'LIN'])]))._initialise()
# ker = testExpr.to_kernel()
# print(get_param_dict(ker))
# print(ker)


def fit_ker_to_kex_with_params(ker, kex, verbose = False):
    if verbose: print(kex)
    param_dict = get_param_dict(ker)
    if verbose: print(param_dict)
    return kex.match_up_fit_parameters(param_dict, 'GP_regression.')


def init_rand_params(kex, verbose = True): # A testing function to initialise a kernel with random parameters
    if verbose: print(kex)
    ker = kex._initialise().to_kernel()
    ker.randomize()
    param_dict = get_param_dict(ker)
    if verbose: print(param_dict)
    return kex.match_up_fit_parameters(param_dict, '')


def remove_top_level_variance(ker):
    has_top_level_variance = 'variance' in ker.parameter_names()
    if has_top_level_variance: ker.unlink_parameter(ker.variance)
    return has_top_level_variance
# # ker = LIN()
# ker = SE()
# # ker = CP(LIN(), WN())
# print(ker.parameter_names())
# print(remove_top_level_variance(ker))
# print(ker.parameter_names())
