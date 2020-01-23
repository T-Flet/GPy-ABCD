from paramz.caching import Cache_this
from paramz.transformations import Logexp
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param

from GPy_ABCD.Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel


class ChangeKernelBase(CombinationKernel):
    """
    Abstract class for change kernels
    """
    def __init__(self, left, right, sigmoidal, location: float = 0., slope: float = 1., name='change_base'):
        _newkerns = [kern.copy() for kern in (left, right)]
        super(ChangeKernelBase, self).__init__(_newkerns, name)
        self.left = left
        self.right = right
        self.slope = Param('slope', slope, Logexp())
        if isinstance(location, tuple):
            self.sigmoidal = sigmoidal(1, False, 1., location[0], location[1], slope)
            self.sigmoidal_reverse = sigmoidal(1, True, 1., location[0], location[1], slope)
            self.location = Param('location', location[0])
            self.stop_location = Param('stop_location', location[1])
            self.link_parameters(self.location, self.stop_location, self.slope)
        else:
            self.sigmoidal = sigmoidal(1, False, 1., location, slope)
            self.sigmoidal_reverse = sigmoidal(1, True, 1., location, slope)
            self.location = Param('location', location)
            self.link_parameters(self.location, self.slope)

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
        return self.left.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.right.K(X, X2) * self.sigmoidal.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.left.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.right.Kdiag(X) * self.sigmoidal.Kdiag(X)

    def update_gradients_full(self, dL_dK, X, X2 = None):
        [p.update_gradients_full(dL_dK, X, X2) for p in [self.left, self.right, self.sigmoidal] if not p.is_fixed]
        self.location.gradient = self.sigmoidal.location.gradient
        self.slope.gradient = self.sigmoidal.slope.gradient

    def update_gradients_diag(self, dL_dK, X):
        [p.update_gradients_diag(dL_dK, X) for p in [self.left, self.right, self.sigmoidal]]
        self.location.gradient = self.sigmoidal.location.gradient
        self.slope.gradient = self.sigmoidal.slope.gradient


class ChangePointKernel(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at location"""
    def __init__(self, left, right, location: float = 0., slope: float = 1., name='change_point'):
        super(ChangePointKernel, self).__init__(left, right, SigmoidalKernel, location, slope, name)


class ChangeWindowKernelOneLocation(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at a limited location"""
    def __init__(self, left, right, location: float = 0., slope: float = 1., name='change_window_one_location'):
        super(ChangeWindowKernelOneLocation, self).__init__(left, right, SigmoidalIndicatorKernel, location, slope, name)


class ChangeWindowKernel(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at a limited location"""
    def __init__(self, left, right, location: float = 0., stop_location: float = 1., slope: float = 1., name='change_window'):
        super(ChangeWindowKernel, self).__init__(left, right, SigmoidalIndicatorKernel, (location, stop_location), slope, name)

    def parameters_changed(self):
        super(ChangeWindowKernel, self).parameters_changed()
        self.sigmoidal_reverse.stop_location = self.sigmoidal.stop_location = self.stop_location

    def update_gradients_full(self, dL_dK, X, X2 = None):
        super(ChangeWindowKernel, self).update_gradients_full(dL_dK, X, X2)
        self.stop_location.gradient = self.sigmoidal.stop_location.gradient

    def update_gradients_diag(self, dL_dK, X):
        super(ChangeWindowKernel, self).update_gradients_diag(dL_dK, X)
        self.stop_location.gradient = self.sigmoidal.stop_location.gradient


class ChangeWindowKernelWithWidth(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at a limited location"""
    def __init__(self, left, right, location: float = 0., slope: float = 1., width: float = 1., name='change_window'):
        super(ChangeWindowKernelWithWidth, self).__init__(left, right, SigmoidalIndicatorKernel, location, slope, name)
        self.width = Param('width', width, Logexp())
        self.link_parameter(self.width)

    def parameters_changed(self):
        super(ChangeWindowKernelWithWidth, self).parameters_changed()
        self.sigmoidal_reverse.width = self.sigmoidal.width = self.width

    def update_gradients_full(self, dL_dK, X, X2 = None):
        super(ChangeWindowKernelWithWidth, self).update_gradients_full(dL_dK, X, X2)
        self.width.gradient = self.sigmoidal.width.gradient

    def update_gradients_diag(self, dL_dK, X):
        super(ChangeWindowKernelWithWidth, self).update_gradients_diag(dL_dK, X)
        self.width.gradient = self.sigmoidal.width.gradient



# TODO:
#   Idea: have only one version of each change operator (NOT the sigmoidal kernels themselves), which can fit the reverse too by different parameter values
#       Pros: have to fit HALF the number of change models
#       Con: the kernel expression does not always match the actual shape anymore
#   Separately: polish the two-location version of the sigmoidal indicator kernel
#
#   NOTE: is_reversed() is unreliable since these are covariances after all, therefore the curve may take one direction or the other
#           Is there a check using X and location(s) in order to be sure?
