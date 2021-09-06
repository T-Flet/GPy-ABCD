import numpy as np
from scipy.special import i0, i1
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

from GPy_ABCD.Util.numba_types import *
from GPy_ABCD.Util.benchmarking import compare_implementations, numba_comparisons


@njit(f8A(f8A))
def _embi0(x): # == exp(-x) * besseli(0, x) => 9.8.2 Abramowitz & Stegun (http://people.math.sfu.ca/~cbm/aands/page_378.htm)
    y = 3.75 / x # The below is an efficient polynomial computation
    f = 0.39894228 + (0.01328592 + (0.00225319 + (-0.00157565 + (0.00916281 + (-0.02057706 + (0.02635537 + (-0.01647633 + (0.00392377)*y)*y)*y)*y)*y)*y)*y)*y
    return f / np.sqrt(x)
@njit(f8A(f8A))
def _embi1(x): # == exp(-x) * besseli(1, x) => 9.8.4 Abramowitz & Stegun (http://people.math.sfu.ca/~cbm/aands/page_378.htm)
    y = 3.75 / x # The below is an efficient polynomial computation
    f = 0.39894228 + (-0.03988024 + (-0.00362018 + (0.00163801 + (-0.01031555 + (0.02282967 + (-0.02895312 + (0.01787654 + (-0.00420059)*y)*y)*y)*y)*y)*y)*y)*y
    return f / np.sqrt(x)
@njit(f8A(f8A))
def _embi0min1(x): # == embi0(x) - embi1(x)
    y = 3.75 / x # The below is an efficient polynomial computation (with the 0-th power term equal to 0)
    f = (0.05316616 + (0.00587337 + (-0.00321366 + (0.01947836 + (-0.04340673 + (0.05530849 + (-0.03435287 + (0.00812436)*y)*y)*y)*y)*y)*y)*y)*y
    return f / np.sqrt(x)


# Based on GPy's StdPeriodic kernel and gaussianprocess.org/gpml/code/matlab/cov/covPeriodicNoDC.m
class PureStdPeriodicKernel(Kern):
    '''
    The standard periodic kernel due to MacKay (1998) can be decomposed into a sum of a periodic and a constant component;
    this kernel is the purely periodic component, as mentioned in:
    Lloyd, James Robert; Duvenaud, David Kristjanson; Grosse, Roger Baker; Tenenbaum, Joshua B.; Ghahramani, Zoubin (2014):
        Automatic construction and natural-language description of nonparametric regression models.
        In. National Conference on Artificial Intelligence, 7/27/2014, pp. 1242–1250.

    Note: because of its strict nature, this kernel will fit accurately only purely periodic data;
        it needs to be paired with bias or other kernels in order to add flexibility

    .. math::

       k(x,y) = \sigma^2 \frac{\exp \left( \frac{\cos(\frac{2\pi}{p} (x - y) )}{l^2} \right) - I_0\left( \frac{1}{l^2} \right)}
                    {\exp \left( \frac{1}{l^2} \right) - I_0\left( \frac{1}{l^2} \right)}
    where I_0 is the modified Bessel function of the first kind of order 0

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\sigma^2` in the formula above
    :type variance: float
    :param period: the period :math:`p`.
    :type period: array or list of the appropriate size (or float if there is only one period parameter)
    :param lengthscale: the lengthscale :math:`\l`.
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type active_dims: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :rtype: Kernel object
    '''

    def __init__(self, input_dim: int, variance: float = 1., period: float = 2.*np.pi, lengthscale: float = 2.*np.pi,
                 active_dims: int = None, name: str = 'pure_std_periodic') -> None:
        super(PureStdPeriodicKernel, self).__init__(input_dim, active_dims, name)

        self.name = name

        if period is not None:
            period = np.asarray(period)
            assert period.size == input_dim, 'bad number of periods'
        else:
            period = 2.*np.pi * np.ones(input_dim)
        if lengthscale is not None:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size == input_dim, 'bad number of lengthscales'
        else:
            lengthscale = 2.*np.pi * np.ones(input_dim)

        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size == 1, 'Variance size must be one'
        self.period = Param('period', period, Logexp())
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())

        self.link_parameters(self.variance, self.period, self.lengthscale)

    def to_dict(self):
        input_dict = super(PureStdPeriodicKernel, self)._save_to_input_dict()
        input_dict['class'] = 'BesselShiftedPeriodic'
        input_dict['variance'] = self.variance.values.tolist()
        input_dict['period'] = self.period.values.tolist()
        input_dict['lengthscale'] = self.lengthscale.values.tolist()
        return input_dict

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        if X2 is None: X2 = X
        cos_term = np.cos((2 * np.pi / self.period) * (X - X2.T))
        invL2 = 1 / self.lengthscale ** 2

        if np.any(self.lengthscale > 1e4): # Limit for l -> infinity
            return self.variance * cos_term
        elif np.any(invL2 < 3.75):
            exp_term = np.exp(cos_term * invL2)
            bessel0 = i0(invL2)
            return self.variance * ((exp_term - bessel0) / (np.exp(invL2) - bessel0)) # The brackets prevent an overflow; want division first
        else:
            exp_term = np.exp((cos_term - 1) * invL2)
            embi0 = _embi0(invL2)
            return self.variance * ((exp_term - embi0) / (1 - embi0)) # The brackets prevent an overflow; want division first
        # invL2 = 1 / self.lengthscale ** 2
        # bessel0 = i0(invL2)
        # return _K(self.variance, self.period, self.lengthscale, invL2, bessel0, X, X if X2 is None else X2)

    def Kdiag(self, X): # Correct; nice and fast
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X, X2 = None):
        if X2 is None: X2 = X
        trig_arg = (2 * np.pi / self.period) * (X - X2.T)
        cos_term = np.cos(trig_arg)
        sin_term = np.sin(trig_arg)
        invL2 = 1 / self.lengthscale ** 2

        if np.any(self.lengthscale > 1e4):  # Limit for l -> infinity
            dK_dV = cos_term # K / V

            dK_dp = (self.variance / self.period) * trig_arg * sin_term

            # This is 0 in the limit, but best to set it to a small non-0 value
            dK_dl = np.empty_like(dL_dK)
            dK_dl[:, :] = 1e-4 / self.lengthscale[0]
        elif np.any(invL2 < 3.75):
            bessel0 = i0(invL2)
            bessel1 = i1(invL2)
            eInvL2 = np.exp(invL2)
            dInvL2_dl = -2 * invL2 / self.lengthscale # == -2 / l^3

            denom = eInvL2 - bessel0
            exp_term = np.exp(cos_term * invL2)
            K_no_Var = (exp_term - bessel0) / denom # == K / V; here just for clarity of further expressions


            dK_dV = K_no_Var

            dK_dp = (self.variance / self.period) * invL2 * trig_arg * sin_term * exp_term / denom

            dK_dl = dInvL2_dl * self.variance * ( (cos_term * exp_term - bessel1) - K_no_Var * (eInvL2 - bessel1) ) / denom
        else:
            embi0 = _embi0(invL2)
            # embi1 = _embi1(invL2)
            # embi0min1 = embi0 - embi1
            embi0min1 = _embi0min1(invL2)
            dInvL2_dl = -2 * invL2 / self.lengthscale # == -2 / l^3

            denom = 1 - embi0
            exp_term = np.exp((cos_term - 1) * invL2)
            K_no_Var = (exp_term - embi0) / denom # == K / V; here just for clarity of further expressions


            dK_dV = K_no_Var

            dK_dp = (self.variance / self.period) * invL2 * trig_arg * sin_term * exp_term / denom # I.e. SAME as the above case at this abstraction level

            dK_dl = dInvL2_dl * self.variance * ( (cos_term - 1) * exp_term + embi0min1 - K_no_Var * embi0min1 ) / denom

        self.variance.gradient = np.sum(dL_dK * dK_dV)
        self.period.gradient = np.sum(dL_dK * dK_dp)
        self.lengthscale.gradient = np.sum(dL_dK * dK_dl)
        # invL2 = 1 / self.lengthscale ** 2
        # bessel0 = i0(invL2)
        # bessel1 = i1(invL2)
        # self.variance.gradient, self.period.gradient, self.lengthscale.gradient = _update_gradients_full(self.variance, self.period, self.lengthscale, invL2, bessel0, bessel1, dL_dK, X, X if X2 is None else X2)


    # Unused functions for a possible version of this class as a subclass of Stationary

    # def K_of_r(self, r):
    #     cos_term = np.cos(2 * np.pi * r / self.period)
    #
    #     if self.lengthscale > 1e4: # Limit for l -> infinity
    #         return self.variance * cos_term
    #     else:
    #         invL2 = np.sum(1 / self.lengthscale ** 2)
    #         exp_term = np.exp(cos_term * invL2)
    #         bessel0 = i0(invL2)
    #         return self.variance * (exp_term - bessel0) / (np.exp(invL2) - bessel0)
    #
    #
    # def dK_dr(self,r):
    #     trig_arg = 2 * np.pi * r / self.period
    #     cos_term = np.cos(trig_arg)
    #     sin_term = np.sin(trig_arg)
    #
    #     if self.lengthscale > 1e4: # Limit for l -> infinity
    #         return - self.variance * (trig_arg / r) * sin_term
    #     else:
    #         invL2 = np.sum(1 / self.lengthscale ** 2)
    #         exp_term = np.exp(cos_term * invL2)
    #         return - self.variance * invL2 * (trig_arg / r) * sin_term * exp_term / (np.exp(invL2) - i0(invL2))



## numba versions of some methods

# from GPy_ABCD.Util.benchmarking import numba_comparisons
# numba_comparisons(_embi0, f8A(f8A), n = 200, args = [x])


# # if X2 is None: X2 = X
# # invL2 = 1 / lengthscale ** 2
# # bessel0 = i0(invL2)
# # @njit(f8A2(f8A, f8A, f8A, f8A, f8A, f8A2, f8A2)) # The numpy version is faster
# def _K(variance, period, lengthscale, invL2, bessel0, X, X2):
#     cos_term = np.cos((2 * np.pi / period) * (X - X2.T))
#
#     if np.any(lengthscale > 1e4): # Limit for l -> infinity
#         return variance * cos_term
#     elif np.any(invL2 < 3.75):
#         exp_term = np.exp(cos_term * invL2)
#         return variance * ((exp_term - bessel0) / (np.exp(invL2) - bessel0)) # The brackets prevent an overflow; want division first
#     else:
#         exp_term = np.exp((cos_term - 1) * invL2)
#         embi0 = _embi0(invL2)
#         return variance * ((exp_term - embi0) / (1 - embi0)) # The brackets prevent an overflow; want division first



# # invL2 = 1 / lengthscale ** 2
# # bessel0 = i0(invL2)
# # bessel1 = i1(invL2)
# # @njit(nTup(f8, f8, f8)(f8A, f8A, f8A, f8A, f8A, f8A, f8A2, f8A2, f8A2)) # The numpy version is faster
# def _update_gradients_full(variance, period, lengthscale, invL2, bessel0, bessel1, dL_dK, X, X2):
#     trig_arg = (2 * np.pi / period) * (X - X2.T)
#     cos_term = np.cos(trig_arg)
#     sin_term = np.sin(trig_arg)
#
#     if np.any(lengthscale > 1e4):  # Limit for l -> infinity
#         dK_dV = cos_term # K / V
#
#         dK_dp = (variance / period) * trig_arg * sin_term
#
#         # This is 0 in the limit, but best to set it to a small non-0 value
#         dK_dl = np.empty_like(dL_dK)
#         dK_dl[:,:] = 1e-4 / lengthscale[0]
#     elif np.any(invL2 < 3.75):
#         eInvL2 = np.exp(invL2)
#         dInvL2_dl = -2 * invL2 / lengthscale # == -2 / l^3
#
#         denom = eInvL2 - bessel0
#         exp_term = np.exp(cos_term * invL2)
#         K_no_Var = (exp_term - bessel0) / denom # == K / V; here just for clarity of further expressions
#
#
#         dK_dV = K_no_Var
#
#         dK_dp = (variance / period) * invL2 * trig_arg * sin_term * exp_term / denom
#
#         dK_dl = dInvL2_dl * variance * ( (cos_term * exp_term - bessel1) - K_no_Var * (eInvL2 - bessel1) ) / denom
#     else:
#         embi0 = _embi0(invL2)
#         # embi1 = _embi1(invL2)
#         # embi0min1 = embi0 - embi1
#         embi0min1 = _embi0min1(invL2)
#         dInvL2_dl = -2 * invL2 / lengthscale # == -2 / l^3
#
#         denom = 1 - embi0
#         exp_term = np.exp((cos_term - 1) * invL2)
#         K_no_Var = (exp_term - embi0) / denom # == K / V; here just for clarity of further expressions
#
#
#         dK_dV = K_no_Var
#
#         dK_dp = (variance / period) * invL2 * trig_arg * sin_term * exp_term / denom # I.e. SAME as the above case at this abstraction level
#
#         dK_dl = dInvL2_dl * variance * ( (cos_term - 1) * exp_term + embi0min1 - K_no_Var * embi0min1 ) / denom
#
#     return np.sum(dL_dK * dK_dV), np.sum(dL_dK * dK_dp), np.sum(dL_dK * dK_dl) # variance.gradient, period.gradient, lengthscale.gradient


