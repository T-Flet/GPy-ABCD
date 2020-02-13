from paramz.caching import Cache_this
from paramz.transformations import Logexp, Logistic
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
from copy import deepcopy

from GPy_ABCD.Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel


class ChangeWindowIndependentBase(CombinationKernel):
    """
    Abstract class for 3-part changewindow kernels
    """
    def __init__(self, first, second, sigmoidal, sigmoidal_indicator, third = None, location: float = 0., slope: float = 0.5, width = 1.,
                 name = 'change_window_independent_base', fixed_slope = False):
        third = deepcopy(first) if third is None else third
        _newkerns = [kern.copy() for kern in (first, second, third)]
        super(ChangeWindowIndependentBase, self).__init__(_newkerns, name)
        self.first = first
        self.second = second
        self.third = third

        self._fixed_slope = fixed_slope # Note: here to be used by subclasses, and changing it from the outside does not link the parameter
        if self._fixed_slope: self.slope = slope
        else:
            self.slope = Param('slope', slope, Logexp())
            self.link_parameter(self.slope)

        self.sigmoidal = sigmoidal(1, False, 1., location, slope)
        self.sigmoidal_reverse = sigmoidal(1, True, 1., location, slope)
        self.sigmoidal_indicator = sigmoidal_indicator(1, False, 1., location, slope, width)
        self.location = Param('location', location)
        self.width = Param('width', width, Logexp())
        self.link_parameters(self.location, self.width)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        Note: It uses the private method _save_to_input_dict of the parent.
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(ChangeWindowIndependentBase, self)._save_to_input_dict()
        input_dict["class"] = str("ChangeKernel")
        return input_dict

    def parameters_changed(self):
        self.sigmoidal_indicator.location = self.sigmoidal_reverse.location = self.location
        self.sigmoidal.location = self.location + self.width
        self.sigmoidal_indicator.slope = self.sigmoidal_reverse.slope = self.sigmoidal.slope = self.slope
        self.sigmoidal_indicator.width = self.sigmoidal_reverse.width = self.sigmoidal.width = self.width

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        return self.first.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.second.K(X, X2) * self.sigmoidal_indicator.K(X, X2) + self.third.K(X, X2) * self.sigmoidal.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.first.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.second.Kdiag(X) * self.sigmoidal_indicator.Kdiag(X) + self.third.Kdiag(X) * self.sigmoidal.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        self.first.update_gradients_full(dL_dK * self.sigmoidal_reverse.K(X, X2), X, X2)
        # self.sigmoidal_reverse.update_gradients_full(dL_dK * self.first.K(X, X2), X, X2)
        self.second.update_gradients_full(dL_dK * self.sigmoidal_indicator.K(X, X2), X, X2)
        self.sigmoidal_indicator.update_gradients_full(dL_dK * self.second.K(X, X2), X, X2)
        self.third.update_gradients_full(dL_dK * self.sigmoidal.K(X, X2), X, X2)
        # self.sigmoidal.update_gradients_full(dL_dK * self.third.K(X, X2), X, X2)

        self.location.gradient = self.sigmoidal_indicator.location.gradient# + self.sigmoidal_reverse.location.gradient + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.width.gradient)
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal_indicator.slope.gradient# + self.sigmoidal_reverse.slope.gradient + self.sigmoidal.slope.gradient
        self.width.gradient = self.sigmoidal_indicator.width.gradient# + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.location.gradient)


    def update_gradients_diag(self, dL_dK, X): # See NOTE ON OPTIMISATION
        self.first.update_gradients_diag(dL_dK * self.sigmoidal_reverse.Kdiag(X), X)
        # self.sigmoidal_reverse.update_gradients_diag(dL_dK * self.first.Kdiag(X), X)
        self.second.update_gradients_diag(dL_dK * self.sigmoidal_indicator.Kdiag(X), X)
        self.sigmoidal_indicator.update_gradients_diag(dL_dK * self.second.Kdiag(X), X)
        self.third.update_gradients_diag(dL_dK * self.sigmoidal.Kdiag(X), X)
        # self.sigmoidal.update_gradients_diag(dL_dK * self.third.Kdiag(X), X)

        self.location.gradient = self.sigmoidal_indicator.location.gradient# + self.sigmoidal_reverse.location.gradient + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.width.gradient)
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal_indicator.slope.gradient# + self.sigmoidal_reverse.slope.gradient + self.sigmoidal.slope.gradient
        self.width.gradient = self.sigmoidal_indicator.width.gradient# + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.location.gradient)


class ChangeWindowKernelIndependent(ChangeWindowIndependentBase):
    """Composite kernel changing from first/third to second subkernels at a limited location and fitting a separate instance of the first one as the third"""
    def __init__(self, first, second, third = None, location: float = 0., slope: float = 0.5, width: float = 1., name='change_window', fixed_slope = False):
        super(ChangeWindowKernelIndependent, self).__init__(first, second, SigmoidalKernel, SigmoidalIndicatorKernel, third, location, slope, width, name, fixed_slope)