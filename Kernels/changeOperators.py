from Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel
from paramz.caching import Cache_this
from GPy.kern.src.kern import CombinationKernel
from GPy.core.parameterization import Param
from paramz.transformations import Logexp


class ChangeKernelBase(CombinationKernel):
    """
    Abstract class for change kernels
    """
    def __init__(self, left, right, sigmoidal, variance: float = 1., location: float = 0., slope: float = 1., name='change_base'):
        _newkerns = [kern.copy() for kern in (left, right)]
        super(ChangeKernelBase, self).__init__(_newkerns, name)
        self.left = left
        self.right = right
        self.sigmoidal = sigmoidal(1, False, variance, location, slope)
        self.sigmoidal_reverse = sigmoidal(1, True, variance, location, slope)
        self.variance = Param('variance', variance, Logexp())
        self.location = Param('location', location)
        self.slope = Param('slope', slope, Logexp()) # Logexp is the default constrain_positive constraint
        self.link_parameters(self.variance, self.location, self.slope)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        Note: It uses the private method _save_to_input_dict of the parent.
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(ChangeKernelBase, self)._save_to_input_dict()
        input_dict["class"] = str("ChangeKernel")
        return input_dict

    @Cache_this(limit=3,)
    def K(self, X, X2 = None):
        self.sigmoidal_reverse.variance = self.sigmoidal.variance = self.variance
        self.sigmoidal_reverse.location = self.sigmoidal.location = self.location
        self.sigmoidal_reverse.slope = self.sigmoidal.slope = self.slope
        return self.left.K(X, X2) * self.sigmoidal_reverse.K(X, X2) + self.right.K(X, X2) * self.sigmoidal.K(X, X2)

    @Cache_this(limit=3)
    def Kdiag(self, X):
        self.sigmoidal_reverse.variance = self.sigmoidal.variance = self.variance
        self.sigmoidal_reverse.location = self.sigmoidal.location = self.location
        self.sigmoidal_reverse.slope = self.sigmoidal.slope = self.slope
        return self.left.Kdiag(X) * self.sigmoidal_reverse.Kdiag(X) + self.right.Kdiag(X) * self.sigmoidal.Kdiag(X)

    def update_gradients_full(self, dL_dK, X, X2 = None):
        [p.update_gradients_full(dL_dK, X, X2) for p in [self.left, self.right, self.sigmoidal] if not p.is_fixed]
        self.variance.gradient = self.sigmoidal.variance.gradient
        self.location.gradient = self.sigmoidal.location.gradient
        self.slope.gradient = self.sigmoidal.slope.gradient

    def update_gradients_diag(self, dL_dK, X):
        [p.update_gradients_diag(dL_dK, X) for p in [self.left, self.right, self.sigmoidal]]
        self.variance.gradient = self.sigmoidal.variance.gradient
        self.location.gradient = self.sigmoidal.location.gradient
        self.slope.gradient = self.sigmoidal.slope.gradient


class ChangePointKernel(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at location"""
    def __init__(self, left, right, variance: float = 1., location: float = 0., slope: float = 1., name='change_point'):
        super(ChangePointKernel, self).__init__(left, right, SigmoidalKernel, variance, location, slope, name)


class ChangeWindowKernel(ChangeKernelBase):
    """Composite kernel changing from left to right subkernels at a limited location"""
    def __init__(self, left, right, variance: float = 1., location: float = 0., slope: float = 1., name='change_window'):
        super(ChangeWindowKernel, self).__init__(left, right, SigmoidalIndicatorKernel, variance, location, slope, name)


# TODO:
#   Commit this version
#   Remove variance
#   Add 2nd location for Changewindow kernel
