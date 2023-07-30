import numpy as np
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

from GPy_ABCD.Util.numba_types import *


class LinearWithOffset(Kern):
    '''
    Linear kernel with horizontal offset

    .. math::

       k(x,y) = \sigma^2 (x - o)(y - o)

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the variance :math:`\sigma^2`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param offset: the horizontal offset :math:`\o`.
    :type offset: array or list of the appropriate size (or float if there is only one offset parameter)
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type active_dims: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :rtype: Kernel object
    '''

    def __init__(self, input_dim: int, variance: float = 1., offset: float = 0., active_dims: int = None, name: str = 'linear_with_offset') -> None:
        super(LinearWithOffset, self).__init__(input_dim, active_dims, name)
        if variance is not None:
            variance = np.asarray(variance)
            assert variance.size == 1
        else:
            variance = np.ones(1)

        self.variance = Param('variance', variance, Logexp())
        self.offset = Param('offset', offset)

        self.link_parameters(self.variance, self.offset)

    def to_dict(self):
        input_dict = super(LinearWithOffset, self)._save_to_input_dict()
        input_dict['class'] = 'LinearWithOffset'
        input_dict['variance'] = self.variance.values.tolist()
        input_dict['offset'] = self.offset
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return LinearWithOffset(**input_dict)

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        if X2 is None: X2 = X
        return self.variance * (X - self.offset) * (X2 - self.offset).T
        # return _K(self.variance, self.offset, X, X if X2 is None else X2)

    def Kdiag(self, X):
        # return np.sum(self.variance * np.square(X - self.offset), -1)
        return _Kdiag(self.variance, self.offset, X)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X
        dK_dV = (X - self.offset) * (X2 - self.offset).T
        dK_do = self.variance * (2 * self.offset - X - X2)

        self.variance.gradient = np.sum(dL_dK * dK_dV)
        self.offset.gradient   = np.sum(dL_dK * dK_do)
        # self.variance.gradient, self.offset.gradient = _update_gradients_full(self.offset, self.variance, dL_dK, X, X if X2 is None else X2)



## numba versions of some methods

# from GPy_ABCD.Util.benchmarking import numba_comparisons

# # @njit(f8A2(f8A, f8A, f8A2, f8A2)) # The numpy version is faster
# def _K(variance, offset, X, X2):
#     return variance * (X - offset) * (X2 - offset).T

@njit(f8A(f8A, f8A, f8A2)) # 2x faster than numpy version
def _Kdiag(variance, offset, X):
    return np.sum(variance * np.square(X - offset), -1)

# # @njit(nTup(f8, f8)(f8A, f8A, f8A2, f8A2, f8A2)) # The numpy version is faster
# def _update_gradients_full(offset, variance, dL_dK, X, X2):
#     dK_dV = (X - offset) * (X2 - offset).T
#     dK_do = variance * (2 * offset - X - X2)
#     return np.sum(dL_dK * dK_dV), np.sum(dL_dK * dK_do) # variance.gradient and offset.gradient


