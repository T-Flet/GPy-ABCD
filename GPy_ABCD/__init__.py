"""GPy-ABCD - Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system"""

__version__ = '1.0.1' # Change it in setup.py too
__author__ = 'Thomas Fletcher <T-Fletcher@outlook.com>'
__all__ = []


from GPy_ABCD.Models.modelSearch import GPModel, explore_model_space, model_search_rounds, standard_start_kernels,\
    production_rules_all, fit_one_model, fit_model_list_not_parallel, fit_model_list_parallel
from GPy_ABCD.KernelExpansion.grammar import standard_start_kernels, production_rules_by_type, production_rules_all
from GPy_ABCD.KernelExpansion.kernelOperations import base_kerns, base_sigmoids


