"""GPy-ABCD - Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system"""

__version__ = '1.0.2' # Change it in setup.py too
__author__ = 'Thomas Fletcher <T-Fletcher@outlook.com>'
__all__ = []


# TODO:
#   - Mention fit_kex, fit_GPy_kern in readme, specifying that the former simplifies expressions


from GPy_ABCD.Models.model import GPModel
from GPy_ABCD.Models.modelSearch import explore_model_space, model_search_rounds,\
    fit_one_model, fit_model_list_not_parallel, fit_model_list_parallel
from GPy_ABCD.Util.modelUtil import BIC, AIC, AICc, fit_kex, fit_GPy_kern, model_printout, GPy_optimisers
from GPy_ABCD.KernelExpansion.grammar import start_kernels, production_rules_by_type, production_rules
from GPy_ABCD.KernelExpansion.kernelOperations import base_kerns, base_sigmoids


