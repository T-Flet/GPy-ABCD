from paramz.caching import Cache_this
from paramz.transformations import Logexp
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
import GPy.kern as _Gk
import numpy as np

from GPy_ABCD.Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel


class ChangeWindowShiftedSidesBase(CombinationKernel):
    """
    Abstract class for changewindow kernels with the two sides being allowed a vertical shift difference
    """
    def __init__(self, first, second, sigmoidal, sigmoidal_indicator, location: float = 0., slope: float = 0.5, width = 1.,
                 name = 'change_window_shifted_sides_base', fixed_slope = False):
        _newkerns = [kern.copy() for kern in (first, second)]
        super(ChangeWindowShiftedSidesBase, self).__init__(_newkerns, name)
        self.first = first
        self.second = second

        self._fixed_slope = fixed_slope # Note: here to be used by subclasses, and changing it from the outside does not link the parameter
        if self._fixed_slope: self.slope = slope
        else:
            self.slope = Param('slope', np.array(slope), Logexp())
            self.link_parameter(self.slope)

        self.sigmoidal = sigmoidal(1, False, 1., location, slope)
        self.sigmoidal_reverse = sigmoidal(1, True, 1., location, slope)
        self.sigmoidal_indicator = sigmoidal_indicator(1, False, 1., location, slope, width)
        # self.shift = _Gk.Bias(1)
        self.location = Param('location', np.array(location))
        self.width = Param('width', np.array(width), Logexp())
        # self.shift_variance = Param('shift_variance', self.shift.variance.values, Logexp())
        self.shift_variance = Param('shift_variance', np.array(0), Logexp())
        self.link_parameters(self.location, self.width, self.shift_variance)

        # self.data_range = None
        # self.one_off_bounds_set = False
        # self.last_parameter_values = {'location': np.array(location), 'slope': np.array(slope), 'width': np.array(width)}

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        Note: It uses the private method _save_to_input_dict of the parent.
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(ChangeWindowShiftedSidesBase, self)._save_to_input_dict()
        input_dict["class"] = str("ChangeKernel")
        return input_dict

    def parameters_changed(self):
        # if np.isnan(self.location): self.location = self.last_parameter_values['location']
        # else: self.last_parameter_values['location'] = np.array(self.location)
        # if np.isnan(self.slope): self.slope = self.last_parameter_values['slope']
        # else: self.last_parameter_values['slope'] = np.array(self.slope)
        # if np.isnan(self.width): self.width = self.last_parameter_values['width']
        # else: self.last_parameter_values['width'] = np.array(self.width)

        self.sigmoidal_indicator.location = self.sigmoidal_reverse.location = self.location
        self.sigmoidal.location = self.location + self.width
        self.sigmoidal_indicator.slope = self.sigmoidal_reverse.slope = self.sigmoidal.slope = self.slope
        self.sigmoidal_indicator.width = self.width
        # self.shift.variance = self.shift_variance

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        # return self.first.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.second.K(X, X2) * self.sigmoidal_indicator.K(X, X2) + (self.first.K(X, X2) + self.shift.K(X, X2)) * self.sigmoidal.K(X, X2)
        return self.first.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.second.K(X, X2) * self.sigmoidal_indicator.K(X, X2) + (self.first.K(X, X2) + self.shift_variance) * self.sigmoidal.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        # return self.first.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.second.Kdiag(X) * self.sigmoidal_indicator.Kdiag(X) + (self.first.Kdiag(X) + self.shift.Kdiag(X)) * self.sigmoidal.Kdiag(X)
        return self.first.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.second.Kdiag(X) * self.sigmoidal_indicator.Kdiag(X) + (self.first.Kdiag(X) + self.shift_variance) * self.sigmoidal.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    # def update_parameter_bounds(self, X):
    #     if self.data_range is None:
    #         self.data_range = (X.min(), X.max())
    #         self.location = Param('location', self.location, Logistic(*self.data_range))
    #         self.sigmoidal_indicator.location = Param('location', self.location, Logistic(*self.data_range))
    #         # self.sigmoidal_reverse.location = Param('location', self.location, Logistic(*self.data_range))
    #         # self.sigmoidal.location = Param('location', self.location + self.width, Logistic(*self.data_range))
    #         # self.location.constrain_bounded(*self.data_range)
    #         # self.sigmoidal_indicator.location.constrain_bounded(*self.data_range)
    #         # # self.sigmoidal_reverse.location.constrain_bounded(*self.data_range)
    #         # # self.sigmoidal.location.constrain_bounded(*self.data_range)
    #
    #     max_width = self.data_range[1] - self.location
    #     max_width = max_width if max_width > 0 else self.data_range[1] - self.data_range[0]
    #     self.width = Param('width', self.width, Logistic(0, max_width))
    #     self.sigmoidal_indicator.width = Param('width', self.width, Logistic(0, max_width))
    #     # self.width.constrain_bounded(0, max_width)
    #     # self.sigmoidal_indicator.width.constrain_bounded(0, max_width)

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        # self.update_parameter_bounds(X)

        self.second.update_gradients_full(dL_dK * self.sigmoidal_indicator.K(X, X2), X, X2)
        self.sigmoidal_indicator.update_gradients_full(dL_dK * self.second.K(X, X2), X, X2)

        self.first.update_gradients_full(dL_dK * (self.sigmoidal_reverse.K(X, X2) + self.sigmoidal.K(X, X2)), X, X2)
        # self.sigmoidal_reverse.update_gradients_full(dL_dK * self.first.K(X, X2), X, X2)

        self.shift_variance.gradient = np.sum(dL_dK * self.sigmoidal.K(X, X2)) # This is for the single variable case
        # self.shift.update_gradients_full(dL_dK * self.sigmoidal.K(X, X2), X, X2)
        # # self.sigmoidal.update_gradients_full(dL_dK * (self.first.K(X, X2) + self.shift.K(X, X2)), X, X2)

        self.location.gradient = self.sigmoidal_indicator.location.gradient# + self.sigmoidal_reverse.location.gradient + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.width.gradient)
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal_indicator.slope.gradient# + self.sigmoidal_reverse.slope.gradient + self.sigmoidal.slope.gradient
        self.width.gradient = self.sigmoidal_indicator.width.gradient# + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.location.gradient)
        # self.shift_variance.gradient = self.shift.variance.gradient


    def update_gradients_diag(self, dL_dK, X): # See NOTE ON OPTIMISATION
        # self.update_parameter_bounds(X)

        self.second.update_gradients_diag(dL_dK * self.sigmoidal_indicator.Kdiag(X), X)
        self.sigmoidal_indicator.update_gradients_full(dL_dK * self.second.Kdiag(X), X)

        self.first.update_gradients_full(dL_dK * (self.sigmoidal_reverse.Kdiag(X) + self.sigmoidal.Kdiag(X)), X)
        # self.sigmoidal_reverse.update_gradients_full(dL_dK * self.first.Kdiag(X), X)

        self.shift_variance.gradient = np.sum(dL_dK * self.sigmoidal.Kdiag(X)) # This is for the single variable case
        # self.shift.update_gradients_full(dL_dK * self.sigmoidal.Kdiag(X), X)
        # # self.sigmoidal.update_gradients_full(dL_dK * (self.first.Kdiag(X) + self.shift.Kdiag(X)), X)

        self.location.gradient = self.sigmoidal_indicator.location.gradient# + self.sigmoidal_reverse.location.gradient + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.width.gradient)
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal_indicator.slope.gradient# + self.sigmoidal_reverse.slope.gradient + self.sigmoidal.slope.gradient
        self.width.gradient = self.sigmoidal_indicator.width.gradient# + (self.sigmoidal.location.gradient - self.sigmoidal_indicator.location.gradient)
        # self.shift_variance.gradient = self.shift.variance.gradient


class ChangeWindowKernelShiftedSides(ChangeWindowShiftedSidesBase):
    """Composite kernel changing from first/third to second subkernels at a limited location and fitting a separate instance of the first one as the third"""
    def __init__(self, first, second, third = None, location: float = 0., slope: float = 0.5, width: float = 1., name='change_window', fixed_slope = False):
        super(ChangeWindowKernelShiftedSides, self).__init__(first, second, SigmoidalKernel, SigmoidalIndicatorKernel, location, slope, width, name, fixed_slope)