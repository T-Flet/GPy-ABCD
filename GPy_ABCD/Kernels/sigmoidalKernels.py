import numpy as np
from abc import abstractmethod
from GPy.kern.src.basis_funcs import BasisFuncKernel
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this



class SigmoidalKernelBase(BasisFuncKernel):
    """
    Abstract superclass for sigmoidal kernels
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.2,
                 active_dims: int = None, name: str = 'sigmoidal_kernel_base', fixed_slope = False) -> None:
        self.reverse = reverse
        super(SigmoidalKernelBase, self).__init__(input_dim, variance, active_dims, False, name)
        # TO REMOVE VARIANCE: comment line above; uncomment below; remove self.variance factors from subclass methods
        # super(BasisFuncKernel, self).__init__(input_dim, active_dims, name)
        # assert self.input_dim == 1, "Basis Function Kernel only implemented for one dimension. Use one kernel per dimension (and add them together) for more dimensions"
        # self.ARD = False
        # self.variance = 1
        self.location = Param('location', location)
        self.link_parameter(self.location)

        self._fixed_slope = fixed_slope # Note: here to be used by subclasses, and changing it from the outside does not link the parameter
        if self._fixed_slope: self.slope = slope
        else:
            self.slope = Param('slope', slope, Logexp()) # This +ve constraint makes non-reverse sigmoids only fit (+ve or -ve) curves going away from 0; similarly for other kernels
            self.link_parameter(self.slope)

    @staticmethod
    def _sigmoid_function(arg): return (1 + arg / (1 + abs(arg))) / 2
    @staticmethod
    def _sigmoid_function_d(arg): return 0.5 / (1 + abs(arg)) ** 2
    # Since arg is usually (x - l) / s, derivatives by the constituent terms are just _sigmoid_function_d(arg) * ...
    #       x: * 1/s
    #       l: * -1/s
    #       s: * -(x-l)/s^2, i.e. * -arg/s
    # When arg is (x - l - w) / (-s) [NOTE THE -s], derivatives by the constituent terms are just _sigmoid_function_d(arg) * ...
    #       x: * -1/s
    #       l: * 1/s
    #       w: * 1/s
    #       s: * (x-l-w)/(-s)^2, i.e. * -argW/s

    @staticmethod
    def _core_sigmoid_function(arg): return arg / (1 + abs(arg))
    # NOTE: This function can be used to compute the maximum height of the sum of opposite sigmoidals - 1 by feeding it arg = w / (2 * s)
    # Let arg = (x - l) / s and argW (x - l - w) / (-s) [NOTE THE -s]
    # This is the height of sig(arg) + sig(argW) - 1 for midpoint x: l + w/2 in both args (which become w/2s)
    # so 2*sig(w/2s) - 1 = the core sigmoid ([-1,1]) on +ve values because of +ve w and s: core_sig(w/2s)
    @staticmethod
    def _inv_core_sigmoid_d(arg): return - 1 / arg ** 2
    # NOTE: This function comes up in computing the derivatives of the scaled (to 1) version of the above sum of sigmoidals
    # Since arg  is always  w / (2 * s), derivatives are just _inv_core_sigmoid_d(arg) * ...
    #       w: * 1/2s
    #       s: * -w/(2s^2), i.e. * -arg/s

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError



# Based on LogisticBasisFuncKernel
class SigmoidalKernel(SigmoidalKernelBase):
    """
    Sigmoidal kernel
    (ascending by default; descending by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.2,
                 active_dims: int = None, name: str = 'sigmoidal', fixed_slope = False) -> None:
        super(SigmoidalKernel, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        asc = self._sigmoid_function((X - self.location) / self.slope)
        return 1 - asc if self.reverse else asc

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalKernel, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None or X is X2:
            phi1 = self.phi(X)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]

            arg = (X - self.location) / self.slope
            sig_d = self._sigmoid_function_d(arg)

            dphi1_dl = sig_d / (-self.slope)
            self.location.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_dl.T)).sum())

            if not self._fixed_slope:
                dphi1_ds = sig_d * (-arg / self.slope)
                self.slope.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_ds.T)).sum())
        else:
            phi1 = self.phi(X)
            phi2 = self.phi(X2)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]

            arg, arg2 = (X - self.location) / self.slope, (X2 - self.location) / self.slope
            sig_d, sig_d2 = self._sigmoid_function_d(arg), self._sigmoid_function_d(arg2)

            dphi1_dl, dphi2_dl = sig_d / (-self.slope), sig_d2 / (-self.slope)
            self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
            self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

            if not self._fixed_slope:
                dphi1_ds = sig_d * (-arg / self.slope)
                dphi2_ds = sig_d2 * (-arg2 / self.slope)
                self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
                self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient



class SigmoidalIndicatorKernel(SigmoidalKernelBase):
    """
    Sigmoidal indicator function kernel with parametrised start location and width:
    ascendingSigmoid(location) + (1 - ascendingSigmoid(stop_location + width)) - 1, i.e. ascendingSigmoid(location) - ascendingSigmoid(location + width)
    (hat if width > 0, and positive-well otherwise; can flip from one to the other by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.2, width: float = 1.,
                 active_dims: int = None, name: str = 'sigmoidal_indicator', fixed_slope = False) -> None:
        super(SigmoidalIndicatorKernel, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameters(self.width)

    @Cache_this(limit = 3)
    def _phi(self, X):
        asc = self._sigmoid_function((X - self.location) / self.slope)
        desc = self._sigmoid_function((X - self.location - self.width) / (-self.slope))
        height = self._core_sigmoid_function(self.width / (2 * self.slope))
        hat = (asc + desc - 1) / height
        return 1 - hat if self.reverse else hat

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalIndicatorKernel, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None: X2 = X

        phi1 = self.phi(X)
        phi2 = self.phi(X2) if X2 is not X else phi1
        if phi1.ndim != 2:
            phi1 = phi1[:, None]
            phi2 = phi2[:, None] if X2 is not X else phi1

        arg, arg2 = (X - self.location) / self.slope, (X2 - self.location) / self.slope
        argW, argW2 = (X - self.location - self.width) / (-self.slope), (X2 - self.location - self.width) / (-self.slope)
        sig_d, sig_d2 = self._sigmoid_function_d(arg), self._sigmoid_function_d(arg2)
        sig_dW, sig_dW2 = self._sigmoid_function_d(argW), self._sigmoid_function_d(argW2)

        numerator1 = self._sigmoid_function(arg) + self._sigmoid_function(argW) - 1 # asc1 + desc1 - 1
        numerator2 = self._sigmoid_function(arg2) + self._sigmoid_function(argW2) - 1 # asc2 + desc2 - 1

        height_arg = self.width / (2 * self.slope)
        inv_height = 1 / self._core_sigmoid_function(height_arg)
        inv_height_d = self._inv_core_sigmoid_d(height_arg)
        dinvheight_dw = inv_height_d / (2 * self.slope)
        dinvheight_ds = inv_height_d * (-height_arg / self.slope)

        # Repeated _sigmoid_function_d reference:
        # Since arg is usually (x - l) / s, derivatives by the constituent terms are just _sigmoid_function_d(arg) * ...
        #       x: * 1/s
        #       l: * -1/s
        #       s: * -(x-l)/s^2, i.e. * -arg/s
        # When arg is (x - l - w) / (-s) [NOTE THE -s], derivatives by the constituent terms are just _sigmoid_function_d(arg) * ...
        #       x: * -1/s
        #       l: * 1/s
        #       w: * 1/s
        #       s: * (x-l-w)/(-s)^2, i.e. * -argW/s

        dphi1_dl = ((- sig_d + sig_dW) / self.slope) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        dphi2_dl = ((- sig_d2 + sig_dW2) / self.slope) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
        self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

        if not self._fixed_slope:
            dphi1_ds = ((sig_d * arg + sig_dW * argW) / (-self.slope)) * inv_height + numerator1 * dinvheight_ds
            dphi2_ds = ((sig_d2 * arg2 + sig_dW2 * argW2) / (-self.slope)) * inv_height + numerator2 * dinvheight_ds
            self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
            self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        dphi1_dw = (sig_dW / self.slope) * inv_height + numerator1 * dinvheight_dw # d(asc)_dw = 0
        dphi2_dw = (sig_dW2 / self.slope) * inv_height + numerator2 * dinvheight_dw # d(asc)_dw = 0
        self.width.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dw.T) + phi2.dot(dphi1_dw.T))).sum())
        self.width.gradient = np.where(np.isnan(self.width.gradient), 0, self.width.gradient)


        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient
            self.width.gradient = - self.width.gradient


