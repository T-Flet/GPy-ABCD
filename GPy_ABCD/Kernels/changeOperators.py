from paramz.caching import Cache_this
from paramz.transformations import Logexp
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
import numpy as np
from copy import copy

from GPy_ABCD.Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel


class ChangeKernelBase(CombinationKernel):
    """
    Abstract class for change kernels
    """
    def __init__(self, first, second, sigmoidal, location: float = 0., slope: float = 0.5, name = 'change_base', fixed_slope = False):
        _newkerns = [kern.copy() for kern in (first, second)]
        super(ChangeKernelBase, self).__init__(_newkerns, name)
        self.first = first
        self.second = second

        self._fixed_slope = fixed_slope # Note: here to be used by subclasses, and changing it from the outside does not link the parameter
        if self._fixed_slope: self.slope = slope
        else:
            self.slope = Param('slope', slope, Logexp())
            self.link_parameter(self.slope)

        if isinstance(location, tuple):
            self.sigmoidal = sigmoidal(1, False, 1., location[0], location[1], slope)
            self.sigmoidal_reverse = sigmoidal(1, True, 1., location[0], location[1], slope)
            self.location = Param('location', location[0])
            self.stop_location = Param('stop_location', location[1])
            self.link_parameters(self.location, self.stop_location)
        else:
            self.sigmoidal = sigmoidal(1, False, 1., location, slope)
            self.sigmoidal_reverse = sigmoidal(1, True, 1., location, slope)
            self.location = Param('location', location)
            self.link_parameter(self.location)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        Note: It uses the private method _save_to_input_dict of the parent.
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(ChangeKernelBase, self)._save_to_input_dict()
        input_dict["class"] = str("ChangeKernel")
        return input_dict

    def parameters_changed(self):
        self.sigmoidal_reverse.location = self.sigmoidal.location = self.location
        self.sigmoidal_reverse.slope = self.sigmoidal.slope = self.slope

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        return self.first.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.second.K(X, X2) * self.sigmoidal.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.first.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.second.Kdiag(X) * self.sigmoidal.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        self.second.update_gradients_full(dL_dK * self.sigmoidal.K(X, X2), X, X2)
        self.sigmoidal.update_gradients_full(dL_dK * self.second.K(X, X2), X, X2)

        self.first.update_gradients_full(dL_dK * self.sigmoidal_reverse.K(X, X2), X, X2)
        # self.sigmoidal_reverse.update_gradients_full(dL_dK * self.first.K(X, X2), X, X2)

        self.location.gradient = self.sigmoidal.location.gradient# + self.sigmoidal_reverse.location.gradient
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal.slope.gradient# + self.sigmoidal_reverse.slope.gradient


    def update_gradients_diag(self, dL_dK, X): # See NOTE ON OPTIMISATION
        self.second.update_gradients_diag(dL_dK * self.sigmoidal.Kdiag(X), X)
        self.sigmoidal.update_gradients_diag(dL_dK * self.second.Kdiag(X), X)

        self.first.update_gradients_diag(dL_dK * self.sigmoidal_reverse.Kdiag(X), X)
        # self.sigmoidal_reverse.update_gradients_diag(dL_dK * self.first.Kdiag(X), X)

        self.location.gradient = self.sigmoidal.location.gradient# + self.sigmoidal_reverse.location.gradient
        if not self._fixed_slope: self.slope.gradient = self.sigmoidal.slope.gradient# + self.sigmoidal_reverse.slope.gradient


class ChangePointKernel(ChangeKernelBase):
    """Composite kernel changing from first to second subkernels at location"""
    def __init__(self, first, second, location: float = 0., slope: float = 0.5, name='change_point', fixed_slope = False):
        super(ChangePointKernel, self).__init__(first, second, SigmoidalKernel, location, slope, name, fixed_slope)


# NOTE: Multiple versions of ChangeWindow Kernels appear below, each using SigmoidalIndicator Kernels with different parameters:
#   - Start location and full width # USING THIS ONE
#       - 'Corrected'-width version of the above (shifting it by the opposite of the location change to preserve the implied 2nd location)
#       - Alternating location-width optimising version of the above
#   - Central location and full width
#   - Just location ("fixed" width term) # This actually uses the 2 location one as well
#   - Start and end locations

# TODO:
#  - make the width update "correctly", i.e. only because of its gradient, therefore adjusting its post-optimisation value by the opposite of the location change (down to 0)
#  - alternate optimsiation of location and width every other `update_gradients`, shifting the width manually when the location changes to compensate
#  - try an almost-3-part changewindow: add a vertical shift C() for the third part

class ChangeWindowKernel(ChangeKernelBase):
    """Composite kernel changing from first to second subkernels at a limited location"""
    def __init__(self, first, second, location: float = 0., slope: float = 0.5, width: float = 1., name='change_window', fixed_slope = False):
        super(ChangeWindowKernel, self).__init__(first, second, SigmoidalIndicatorKernel, location, slope, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameter(self.width)

    def parameters_changed(self):
        super(ChangeWindowKernel, self).parameters_changed()
        self.sigmoidal_reverse.width = self.sigmoidal.width = self.width

    def update_gradients_full(self, dL_dK, X, X2 = None):
        super(ChangeWindowKernel, self).update_gradients_full(dL_dK, X, X2)
        self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase

    def update_gradients_diag(self, dL_dK, X):
        super(ChangeWindowKernel, self).update_gradients_diag(dL_dK, X)
        self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase

class ChangeWindowKernelCorrectedWidth(ChangeKernelBase):
    """Composite kernel changing from first to second subkernels at a limited location"""
    def __init__(self, first, second, location: float = 0., slope: float = 0.5, width: float = 1., name='change_window', fixed_slope = False):
        super(ChangeWindowKernelCorrectedWidth, self).__init__(first, second, SigmoidalIndicatorKernel, location, slope, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameter(self.width)
        self.old_location = self.location.values

    def parameters_changed(self):
        super(ChangeWindowKernelCorrectedWidth, self).parameters_changed()
        location_diff = self.location - self.old_location
        # self.width = 1e-10 if location_diff > self.width else self.width - location_diff # THIS GENERATES AN INFINITE LOOP
        # self.sigmoidal_reverse.width = self.sigmoidal.width = self.width
        self.sigmoidal_reverse.width = self.sigmoidal.width = 1e-10 if location_diff > self.width else self.width - location_diff

    def update_gradients_full(self, dL_dK, X, X2 = None):
        super(ChangeWindowKernelCorrectedWidth, self).update_gradients_full(dL_dK, X, X2)
        self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase
        self.old_location = self.location.values

    def update_gradients_diag(self, dL_dK, X):
        super(ChangeWindowKernelCorrectedWidth, self).update_gradients_diag(dL_dK, X)
        self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase
        self.old_location = self.location.values

class ChangeWindowKernelAlternating(ChangeKernelBase):
    """Composite kernel changing from first to second subkernels at a limited location"""
    def __init__(self, first, second, location: float = 0., slope: float = 0.5, width: float = 1., name='change_window', fixed_slope = False):
        super(ChangeWindowKernelAlternating, self).__init__(first, second, SigmoidalIndicatorKernel, location, slope, name, fixed_slope)
        self.width = Param('width', width, Logexp())
        self.link_parameter(self.width)
        self.alternator = False
        self.old_location = copy(self.location)

    def parameters_changed(self):
        super(ChangeWindowKernelAlternating, self).parameters_changed()
        if self.alternator:
            location_diff = self.location - self.old_location
            self.sigmoidal_reverse.width = self.sigmoidal.width = self.width = 1e-10 if location_diff > self.width else self.width - location_diff # THIS GENERATES AN INFINITE LOOP
        else: self.sigmoidal_reverse.width = self.sigmoidal.width = self.width
        # self.sigmoidal_reverse.width = self.sigmoidal.width = self.width

    def update_gradients_full(self, dL_dK, X, X2 = None):
        super(ChangeWindowKernelAlternating, self).update_gradients_full(dL_dK, X, X2)
        self.alternator = not self.alternator
        if self.alternator:
            self.width.gradient = np.array(0)
            self.old_location = copy(self.location)
        else:
            self.location.gradient = np.array(0)
            self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase

    def update_gradients_diag(self, dL_dK, X):
        super(ChangeWindowKernelAlternating, self).update_gradients_diag(dL_dK, X)
        self.alternator = not self.alternator
        if self.alternator:
            self.width.gradient = np.array(0)
            self.old_location = copy(self.location)
        else:
            self.location.gradient = np.array(0)
            self.width.gradient = self.sigmoidal.width.gradient# + self.sigmoidal_reverse.width.gradient # See NOTE ON OPTIMISATION in ChangeKernelBase



# TODO:
#   Add a 3-part changewindow operator (with separate instances of first and second kernels) (maybe to be used conditionally
#       on whether first/second is non-stationary)
#   Idea: have only one version of each change operator (NOT the sigmoidal kernels themselves), which can fit the reverse too by different parameter values
#       Pros: have to fit HALF the number of change models
#       Con: the kernel expression does not always match the actual shape anymore
#   Separately: polish the two-location version of the sigmoidal indicator kernel


