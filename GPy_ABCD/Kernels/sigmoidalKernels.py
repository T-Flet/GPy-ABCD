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

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.5,
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

    # NOTE: is_reversed() is NOT always reliable since these are only covariances after all and therefore flipped-direction fits can just happen
    #   HOWEVER: when it works this is "reversed" w.r.t. whichever value self.reverse has
    @abstractmethod
    def is_reversed(self):
        pass

    @staticmethod
    def _sech(x): # Because the np.cosh overflows too easily before inversion
        safe_x = np.where(np.abs(x) < 700, x, np.sign(x) * 700) # 709.782712893384 is the real np.log(1.7976931348623157e+308)
        return 2 / (np.exp(safe_x) + np.exp(-safe_x))
    @staticmethod
    def _csch(x):
        safe_x = np.where(np.abs(x) < 700, x, np.sign(x) * 700) # 709.782712893384 is the real np.log(1.7976931348623157e+308)
        denom = np.exp(safe_x) - np.exp(-safe_x)
        return 2 / denom if denom != 0 else np.sign(safe_x) * 1e30
    @staticmethod
    def _sigmoid_function(x, l, s):
        return (1 + np.tanh( (x - l) / s )) / 2
    @staticmethod
    def _sigmoid_function_dl(x, l, s):
        minus_arg = (l - x) / s
        return - SigmoidalKernelBase._sech(minus_arg) ** 2 / (2 * s)
    @staticmethod
    def _sigmoid_function_ds(x, l, s):
        minus_arg = (l - x) / s
        return minus_arg * SigmoidalKernelBase._sech(minus_arg) ** 2 / (2 * s)

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError



# Based on LogisticBasisFuncKernel
class SigmoidalKernel(SigmoidalKernelBase):
    """
    Sigmoidal kernel
    (ascending by default; descending by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.5,
                 active_dims: int = None, name: str = 'sigmoidal', fixed_slope = False) -> None:
        super(SigmoidalKernel, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)

    def is_reversed(self):
        return (self.slope < 0) ^ self.reverse

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        asc = self._sigmoid_function(X, self.location, self.slope)
        return 1 - asc if self.reverse else asc

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalKernel, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None or X is X2:
            phi1 = self.phi(X)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]

            dphi1_dl = self._sigmoid_function_dl(X, self.location, self.slope)
            self.location.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_dl.T)).sum())

            if not self._fixed_slope:
                dphi1_ds = self._sigmoid_function_ds(X, self.location, self.slope)
                self.slope.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_ds.T)).sum())
        else:
            phi1 = self.phi(X)
            phi2 = self.phi(X2)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]

            dphi1_dl = self._sigmoid_function_dl(X, self.location, self.slope)
            dphi2_dl = self._sigmoid_function_dl(X2, self.location, self.slope)
            self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
            self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

            if not self._fixed_slope:
                dphi1_ds = self._sigmoid_function_ds(X, self.location, self.slope)
                dphi2_ds = self._sigmoid_function_ds(X2, self.location, self.slope)
                self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
                self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient


# NOTE: Multiple versions of SigmoidalIndicator Kernels appear below, each with different parameters:
#   - Start location and full width # USING THIS ONE
#   - Central location and full width
#   - Just location ("fixed" width term)
#   - Start and end locations


class SigmoidalIndicatorKernel(SigmoidalKernelBase):
    """
    Sigmoidal indicator function kernel with parametrised start location and width:
    ascendingSigmoid(location) + (1 - ascendingSigmoid(stop_location + width)) - 1, i.e. ascendingSigmoid(location) - ascendingSigmoid(location + width)
    (hat if width > 0, and positive-well otherwise; can flip from one to the other by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.5, width: float = 1.,
                 active_dims: int = None, name: str = 'sigmoidal_indicator', fixed_slope = False) -> None:
        super(SigmoidalIndicatorKernel, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameters(self.width)

    @staticmethod
    def _sigmoid_function_dw(x, l, s, w): # 0 if no width or same as dl one but with l -> l + w
        # return 0 if w is None else SigmoidalKernelBase._sigmoid_function_dl(x, l + w, s) # Here just for clarity; the 0 case is just omitted later
        return SigmoidalKernelBase._sigmoid_function_dl(x, l + w, s)

    @staticmethod
    def _sigmoid_function_opposites_sum_height(w, s): # Height of sig(x, l, s) + sig(x, l + w, -s) - 1, i.e. the value at x = l + w/2
        return np.tanh(w / (2 * s))

    def is_reversed(self):
        return (self.slope < 0) ^ self.reverse

    @Cache_this(limit = 3)
    def _phi(self, X):
        asc = self._sigmoid_function(X, self.location, self.slope)
        desc = self._sigmoid_function(X, self.location + self.width, -self.slope)
        height = SigmoidalIndicatorKernel._sigmoid_function_opposites_sum_height(self.width, self.slope)
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

        inv_height = 1 / SigmoidalIndicatorKernel._sigmoid_function_opposites_sum_height(self.width, self.slope)
        numerator1 = self._sigmoid_function(X, self.location, self.slope) + self._sigmoid_function(X, self.location + self.width, -self.slope) - 1 # asc1 + desc1 - 1
        numerator2 = self._sigmoid_function(X2, self.location, self.slope) + self._sigmoid_function(X2, self.location + self.width, -self.slope) - 1 # asc2 + desc2 - 1

        height_arg = self.width / (2 * self.slope)
        dinvheight_dw = - (SigmoidalKernelBase._csch(height_arg) ** 2) / (2 * self.slope)
        dinvheight_ds = - 2 * height_arg * dinvheight_dw

        dphi1_dl = (self._sigmoid_function_dl(X, self.location, self.slope) + self._sigmoid_function_dl(X, self.location + self.width, -self.slope)) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        dphi2_dl = (self._sigmoid_function_dl(X2, self.location, self.slope) + self._sigmoid_function_dl(X2, self.location + self.width, -self.slope)) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
        self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

        if not self._fixed_slope:
            dphi1_ds = (self._sigmoid_function_ds(X, self.location, self.slope) - self._sigmoid_function_ds(X, self.location + self.width, -self.slope)) * inv_height + numerator1 * dinvheight_ds
            dphi2_ds = (self._sigmoid_function_ds(X2, self.location, self.slope) - self._sigmoid_function_ds(X2, self.location + self.width, -self.slope)) * inv_height + numerator2 * dinvheight_ds
            self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
            self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        dphi1_dw = self._sigmoid_function_dw(X, self.location, -self.slope, self.width) * inv_height + numerator1 * dinvheight_dw # d(asc)_dw = 0
        dphi2_dw = self._sigmoid_function_dw(X2, self.location, -self.slope, self.width) * inv_height + numerator2 * dinvheight_dw # d(asc)_dw = 0
        self.width.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dw.T) + phi2.dot(dphi1_dw.T))).sum())
        self.width.gradient = np.where(np.isnan(self.width.gradient), 0, self.width.gradient)


        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient
            self.width.gradient = - self.width.gradient


class SigmoidalIndicatorKernelCentreWidth(SigmoidalKernelBase):
    """
    Sigmoidal indicator function kernel with a location and a specific width:
    ascendingSigmoid(location - width/2) + (1 - ascendingSigmoid(stop_location + width/2)) - 1, i.e. ascendingSigmoid(location - width/2) - ascendingSigmoid(location + width/2)
    (hat if width > 0, and positive-well otherwise; can flip from one to the other by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.5, width: float = 1.,
                 active_dims: int = None, name: str = 'sigmoidal_indicator', fixed_slope = False) -> None:
        super(SigmoidalIndicatorKernelCentreWidth, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameters(self.width)

    @staticmethod
    def _sigmoid_function_dw(x, l, s, hw, hw_sign): # plus: original_dl_one(l -> l + w/2) * 1/2; minus: original_dl_one(x -> -x, l -> -l + w/2) * -1/2
        return SigmoidalKernelBase._sigmoid_function_dl(x, l + hw, s) / 2 if hw_sign == 1 else - SigmoidalKernelBase._sigmoid_function_dl(-x, - l + hw, s) / 2

    @staticmethod
    def _sigmoid_function_opposites_sum_height(w, s):  # Height of sig(x, l - w/2, s) + sig(x, l + w/2, -s) - 1
        return np.tanh(w / (2 * s))  # Simplification of: 2 * SigmoidalKernelBase._sigmoid_function(w/2, 0, s) - 1

    def is_reversed(self):
        return ((self.width < 0) ^ (self.slope < 0)) ^ self.reverse

    @Cache_this(limit = 3)
    def _phi(self, X):
        hw = self.width / 2
        asc = self._sigmoid_function(X, self.location - hw, self.slope)
        desc = self._sigmoid_function(X, self.location + hw, -self.slope)
        height = SigmoidalIndicatorKernelCentreWidth._sigmoid_function_opposites_sum_height(self.width, self.slope)
        hat = (asc + desc - 1) / height
        return 1 - hat if self.reverse else hat

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalIndicatorKernelCentreWidth, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None: X2 = X

        phi1 = self.phi(X)
        phi2 = self.phi(X2) if X2 is not X else phi1
        if phi1.ndim != 2:
            phi1 = phi1[:, None]
            phi2 = phi2[:, None] if X2 is not X else phi1

        hw = self.width / 2
        inv_height = 1 / SigmoidalIndicatorKernelCentreWidth._sigmoid_function_opposites_sum_height(self.width, self.slope)
        numerator1 = self._sigmoid_function(X, self.location - hw, self.slope) + self._sigmoid_function(X, self.location + hw, -self.slope) - 1 # asc1 + desc1 - 1
        numerator2 = self._sigmoid_function(X2, self.location - hw, self.slope) + self._sigmoid_function(X2, self.location + hw, -self.slope) - 1 # asc2 + desc2 - 1

        height_arg = self.width / (2 * self.slope)
        dinvheight_dw = - (SigmoidalKernelBase._csch(height_arg) ** 2) / (2 * self.slope)
        dinvheight_ds = - 2 * height_arg * dinvheight_dw

        dphi1_dl = (self._sigmoid_function_dl(X, self.location - hw, self.slope) + self._sigmoid_function_dl(X, self.location + hw, -self.slope)) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        dphi2_dl = (self._sigmoid_function_dl(X2, self.location - hw, self.slope) + self._sigmoid_function_dl(X2, self.location + hw, -self.slope)) * inv_height #+ (asc + desc) * dinvheight_dl, which is 0
        self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
        self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

        if not self._fixed_slope:
            dphi1_ds = (self._sigmoid_function_ds(X, self.location - hw, self.slope) - self._sigmoid_function_ds(X, self.location + hw, -self.slope)) * inv_height + numerator1 * dinvheight_ds
            dphi2_ds = (self._sigmoid_function_ds(X2, self.location - hw, self.slope) - self._sigmoid_function_ds(X2, self.location + hw, -self.slope)) * inv_height + numerator2 * dinvheight_ds
            self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
            self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        dphi1_dw = (self._sigmoid_function_dw(X, self.location, self.slope, hw, -1) + self._sigmoid_function_dw(X, self.location, -self.slope, hw, +1)) * inv_height + numerator1 * dinvheight_dw
        dphi2_dw = (self._sigmoid_function_dw(X2, self.location, self.slope, hw, -1) + self._sigmoid_function_dw(X2, self.location, -self.slope, hw, +1)) * inv_height + numerator2 * dinvheight_dw
        self.width.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dw.T) + phi2.dot(dphi1_dw.T))).sum())
        self.width.gradient = np.where(np.isnan(self.width.gradient), 0, self.width.gradient)


        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient
            self.width.gradient = - self.width.gradient


class SigmoidalIndicatorKernelOneLocation(SigmoidalKernelBase):
    """
    Sigmoidal indicator function kernel with a single location for the centre of the hat/well
    (hat by default; positive-well by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., slope: float = 0.5,
                 active_dims: int = None, name: str = 'sigmoidal_indicator', fixed_slope = False) -> None:
        super(SigmoidalIndicatorKernelOneLocation, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)

    @staticmethod
    def _tanhSigOneLocIndicatorHalfWidth(s, y = 0.01): # Distance from peak of 1-location indicator kernel to when y is reached
        return s * np.arccosh(1 / np.sqrt(y))

    def start_and_end_locations(self, y = 0.5): # Start and end locations for a 1-location indicator kernel
        w = SigmoidalIndicatorKernelOneLocation._tanhSigOneLocIndicatorHalfWidth(self.slope, y)
        return self.location - w, self.location + w

    def is_reversed(self):
        return (self.slope < 0) ^ self.reverse

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        asc = self._sigmoid_function(X, self.location, self.slope)
        hat = 4 * asc * (1 - asc) # Increasing the peak from 0.25 to 1
        return 1 - hat if self.reverse else hat

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalIndicatorKernelOneLocation, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None or X is X2:
            phi1 = self.phi(X)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]

            asc1 = self._sigmoid_function(X, self.location, self.slope)

            dphi1_dl = 4 * (1 - 2 * asc1) * self._sigmoid_function_dl(X, self.location, self.slope)
            self.location.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_dl.T)).sum())

            if not self._fixed_slope:
                dphi1_ds = 4 * (1 - 2 * asc1) * self._sigmoid_function_ds(X, self.location, self.slope)
                self.slope.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_ds.T)).sum())
        else:
            phi1 = self.phi(X)
            phi2 = self.phi(X2)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]

            asc1 = self._sigmoid_function(X, self.location, self.slope)
            asc2 = self._sigmoid_function(X2, self.location, self.slope)

            dphi1_dl = 4 * (1 - 2 * asc1) * self._sigmoid_function_dl(X, self.location, self.slope)
            dphi2_dl = 4 * (1 - 2 * asc2) * self._sigmoid_function_dl(X2, self.location, self.slope)
            self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
            self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

            if not self._fixed_slope:
                dphi1_ds = 4 * (1 - 2 * asc1) * self._sigmoid_function_ds(X, self.location, self.slope)
                dphi2_ds = 4 * (1 - 2 * asc2) * self._sigmoid_function_ds(X2, self.location, self.slope)
                self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
                self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient


class SigmoidalIndicatorKernelTwoLocations(SigmoidalKernelBase):
    """
    Sigmoidal indicator function kernel with a start and a stop location:
    ascendingSigmoid(location) + (1 - ascendingSigmoid(stop_location)) - 1, i.e. ascendingSigmoid(location) - ascendingSigmoid(stop_location)
    (hat if location <= stop_location, and positive-well otherwise; can flip from one to the other by reverse=True)
    """

    def __init__(self, input_dim: int, reverse: bool = False, variance: float = 1., location: float = 0., stop_location: float = 1., slope: float = 0.5,
                 active_dims: int = None, name: str = 'sigmoidal_indicator', fixed_slope = False) -> None:
        super(SigmoidalIndicatorKernelTwoLocations, self).__init__(input_dim, reverse, variance, location, slope, active_dims, name, fixed_slope)
        self.stop_location = Param('stop_location', stop_location)
        self.link_parameters(self.stop_location)

    @staticmethod
    def _sigmoid_function_opposites_sum_height(l, sl, s):  # Height of sig(x, l, s) + sig(x, sl, -s) - 1
        return np.tanh((sl - l) / (2 * s))  # Simplification of: 2 * SigmoidalKernelBase._sigmoid_function((sl - l) / 2, 0, s) - 1

    def is_reversed(self):
        return ((self.location > self.stop_location) ^ (self.slope < 0)) ^ self.reverse

    @Cache_this(limit=3)
    def _phi(self, X):
        asc = self._sigmoid_function(X, self.location, self.slope)
        desc = self._sigmoid_function(X, self.stop_location, -self.slope)
        height = SigmoidalIndicatorKernelTwoLocations._sigmoid_function_opposites_sum_height(self.location, self.stop_location, self.slope)
        hat = (asc + desc - 1) / height #if self.location > self.stop_location else asc - asc2 + 1 # Note: it does not affect the gradients
        return 1 - hat if self.reverse else hat

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(SigmoidalIndicatorKernelTwoLocations, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None: X2 = X

        phi1 = self.phi(X)
        phi2 = self.phi(X2) if X2 is not X else phi1
        if phi1.ndim != 2:
            phi1 = phi1[:, None]
            phi2 = phi2[:, None] if X2 is not X else phi1

        asc1 = self._sigmoid_function(X, self.location, self.slope)
        asc2 = self._sigmoid_function(X2, self.location, self.slope)
        desc1 = self._sigmoid_function(X, self.stop_location, -self.slope)
        desc2 = self._sigmoid_function(X2, self.stop_location, -self.slope)
        inv_height = 1 / SigmoidalIndicatorKernelTwoLocations._sigmoid_function_opposites_sum_height(self.location, self.stop_location, self.slope)
        numerator1 = asc1 + desc1 - 1
        numerator2 = asc2 + desc2 - 1

        height_arg = (self.location - self.stop_location) / (2 * self.slope)
        dinvheight_dl = (SigmoidalKernelBase._csch(height_arg) ** 2) / (2 * self.slope)
        dinvheight_dsl = - dinvheight_dl
        dinvheight_ds = 2 * height_arg * dinvheight_dsl

        # Only asc * inv_height contains l
        dphi1_dl = self._sigmoid_function_dl(X, self.location, self.slope) * inv_height + asc1 * dinvheight_dl
        dphi2_dl = self._sigmoid_function_dl(X2, self.location, self.slope) * inv_height + asc2 * dinvheight_dl
        self.location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dl.T) + phi2.dot(dphi1_dl.T))).sum())
        self.location.gradient = np.where(np.isnan(self.location.gradient), 0, self.location.gradient)

        # Only desc * inv_height contains sl
        dphi1_dsl = self._sigmoid_function_dl(X, self.stop_location, -self.slope) * inv_height + desc1 * dinvheight_dsl
        dphi2_dsl = self._sigmoid_function_dl(X2, self.stop_location, -self.slope) * inv_height + desc2 * dinvheight_dsl
        self.stop_location.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_dsl.T) + phi2.dot(dphi1_dsl.T))).sum())
        self.stop_location.gradient = np.where(np.isnan(self.stop_location.gradient), 0, self.stop_location.gradient)

        if not self._fixed_slope:
            dphi1_ds = (self._sigmoid_function_ds(X, self.location, self.slope) - self._sigmoid_function_ds(X, self.stop_location, -self.slope)) * inv_height + numerator1 * dinvheight_ds
            dphi2_ds = (self._sigmoid_function_ds(X2, self.location, self.slope) - self._sigmoid_function_ds(X2, self.stop_location, -self.slope)) * inv_height + numerator2 * dinvheight_ds
            self.slope.gradient = np.sum(self.variance * (dL_dK * (phi1.dot(dphi2_ds.T) + phi2.dot(dphi1_ds.T))).sum())
            self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)

        if self.reverse: # Works this simply (a single sign flip per product above)
            self.location.gradient = - self.location.gradient
            self.stop_location.gradient = - self.stop_location.gradient
            if not self._fixed_slope: self.slope.gradient = - self.slope.gradient
