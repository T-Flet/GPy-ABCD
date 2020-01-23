import numpy as np
from scipy.special import i0, i1
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


# Based on StdPeriodic and gaussianprocess.org/gpml/code/matlab/cov/covPeriodicNoDC.m
class PureStdPeriodicKernel(Kern):
    """
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
    """

    def __init__(self, input_dim: int, variance: float = 1., period: float = 2.*np.pi, lengthscale: float = 2.*np.pi,
                 active_dims: int = None, name: str = 'pure_std_periodic') -> None:
        super(PureStdPeriodicKernel, self).__init__(input_dim, active_dims, name)

        self.name = name

        if period is not None:
            period = np.asarray(period)
            assert period.size == input_dim, "bad number of periods"
        else:
            period = 2.*np.pi * np.ones(input_dim)
        if lengthscale is not None:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size == input_dim, "bad number of lengthscales"
        else:
            lengthscale = 2.*np.pi * np.ones(input_dim)

        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size == 1, "Variance size must be one"
        self.period = Param('period', period, Logexp())
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())

        self.link_parameters(self.variance, self.period, self.lengthscale)


    def to_dict(self):
        input_dict = super(PureStdPeriodicKernel, self)._save_to_input_dict()
        input_dict["class"] = "BesselShiftedPeriodic"
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["period"] = self.period.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        return input_dict


    @Cache_this(limit=3)
    def K(self, X, X2=None):
        if X2 is None: X2 = X
        cos_term = np.cos(2 * np.pi * (X - X2.T) / self.period)

        if np.any(self.lengthscale > 1e4): # Limit for l -> infinity
            return self.variance * cos_term
        else:
            # Overflow on exp and i0 by going backwards from sys.float_info.max (1.7976931348623157e+308): 1/l^2 < 709.782712893384
            invL2 = np.clip(1 / self.lengthscale ** 2, 0, 700)
            exp_term = np.exp(cos_term * invL2)
            bessel0 = i0(invL2)
            return self.variance * ((exp_term - bessel0) / (np.exp(invL2) - bessel0)) # The brackets prevent an overflow; want division first


    def Kdiag(self, X): # Correct; nice and fast
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret


    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X
        trig_arg = 2 * np.pi * (X - X2.T) / self.period
        cos_term = np.cos(trig_arg)
        sin_term = np.sin(trig_arg)

        if np.any(self.lengthscale > 1e4):  # Limit for l -> infinity
            dK_dV = cos_term # K / V

            dK_dp = self.variance * trig_arg * sin_term / self.period

            # This is 0 in the limit, but best to set it to a small non-0 value
            dK_dl = 1e-4 / self.lengthscale
        else:
            # Overflow on exp and i0 by going backwards from sys.float_info.max (1.7976931348623157e+308): 1/l^2 < 709.782712893384
            invL2 = np.clip(1 / self.lengthscale ** 2, 0, 700)
            bessel0 = i0(invL2)
            bessel1 = i1(invL2)
            eInvL2 = np.exp(invL2)
            denom = eInvL2 - bessel0
            exp_term = np.exp(cos_term * invL2)

            dK_dV = (exp_term - bessel0) / denom # = K / V

            dK_dp = (self.variance * invL2 * trig_arg * sin_term / self.period) * (exp_term / denom) # exp terms division separate because of overflow risk

            K = self.variance * dK_dV
            dInvL2_dl = -2 / self.lengthscale ** 3
            dK_dl = dInvL2_dl * ( self.variance * ((cos_term * exp_term - bessel1) / denom) - K * ((eInvL2 - bessel1) / denom) ) # The brackets prevent some overflows; want those divisions first

        self.variance.gradient = np.sum(dL_dK * dK_dV)
        self.period.gradient = np.sum(dL_dK * dK_dp)
        self.lengthscale.gradient = np.sum(dL_dK * dK_dl)



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
