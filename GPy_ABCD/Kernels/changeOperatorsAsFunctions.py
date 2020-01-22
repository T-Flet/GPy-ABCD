from GPy_ABCD.Kernels.sigmoidalKernels import *


# NOTE: This is all because tie_to is documented but not implemented in GPy yet


def kCP(k1, k2):
    """
    :param k1: Kernel before the changepoint
    :param k2: Kernel after the changepoint
    :return: A kernel composed of k1 being replaced by k2 at some changepoint
    """
    res = k1 * SigmoidalKernel(1, True) + k2 * SigmoidalKernel(1, False)
    res.mul_1.sigmoidal.unlink_parameter(res.mul_1.sigmoidal.variance)
    res.mul_1.sigmoidal.variance = res.mul.sigmoidal.variance
    res.mul_1.sigmoidal.unlink_parameter(res.mul_1.sigmoidal.location)
    res.mul_1.sigmoidal.location = res.mul.sigmoidal.location
    res.mul_1.sigmoidal.unlink_parameter(res.mul_1.sigmoidal.slope)
    res.mul_1.sigmoidal.slope = res.mul.sigmoidal.slope
    return res


def kCW(k1, k2):
    """
    :param k1: Kernel before and after the window
    :param k2: Kernel during the window
    :return: A kernel of k1 replaced by k2 in some changewindow
    """
    res = k1 * SigmoidalIndicatorKernelWithWidth(1, True) + k2 * SigmoidalIndicatorKernelWithWidth(1, False)
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.variance)
    res.mul_1.sigmoidal_indicator.variance = res.mul.sigmoidal_indicator.variance
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.location)
    res.mul_1.sigmoidal_indicator.location = res.mul.sigmoidal_indicator.location
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.width)
    res.mul_1.sigmoidal_indicator.width = res.mul.sigmoidal_indicator.width
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.slope)
    res.mul_1.sigmoidal_indicator.slope = res.mul.sigmoidal_indicator.slope
    return res


def kCW_separate_locs(k1, k2):
    """
    :param k1: Kernel before and after the window
    :param k2: Kernel during the window
    :return: A kernel of k1 replaced by k2 in some changewindow
    """
    res = k1 * SigmoidalIndicatorKernelOneLocation(1, True) + k2 * SigmoidalIndicatorKernelOneLocation(1, False)
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.variance)
    res.mul_1.sigmoidal_indicator.variance = res.mul.sigmoidal_indicator.variance
    # res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.location)
    # res.mul_1.sigmoidal_indicator.location = res.mul.sigmoidal_indicator.location
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.slope)
    res.mul_1.sigmoidal_indicator.slope = res.mul.sigmoidal_indicator.slope
    return res


def kCW_two_locs(k1, k2):
    """
    :param k1: Kernel before and after the window
    :param k2: Kernel during the window
    :return: A kernel of k1 replaced by k2 in some changewindow
    """
    res = k1 * SigmoidalIndicatorKernel(1, True) + k2 * SigmoidalIndicatorKernel(1, False)
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.variance)
    res.mul_1.sigmoidal_indicator.variance = res.mul.sigmoidal_indicator.variance
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.location)
    res.mul_1.sigmoidal_indicator.location = res.mul.sigmoidal_indicator.location
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.stop_location)
    res.mul_1.sigmoidal_indicator.stop_location = res.mul.sigmoidal_indicator.stop_location
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.slope)
    res.mul_1.sigmoidal_indicator.slope = res.mul.sigmoidal_indicator.slope
    return res
