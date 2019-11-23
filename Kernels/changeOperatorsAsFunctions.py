from Kernels.sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernelOneLocation


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
    res = k1 * SigmoidalIndicatorKernelOneLocation(1, True) + k2 * SigmoidalIndicatorKernelOneLocation(1, False)
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.variance)
    res.mul_1.sigmoidal_indicator.variance = res.mul.sigmoidal_indicator.variance
    # res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.location)
    # res.mul_1.sigmoidal_indicator.location = res.mul.sigmoidal_indicator.location
    res.mul_1.sigmoidal_indicator.unlink_parameter(res.mul_1.sigmoidal_indicator.slope)
    res.mul_1.sigmoidal_indicator.slope = res.mul.sigmoidal_indicator.slope
    return res


# TODO:
#   NOTE: This is all because tie_to is documented but not implemented in GPy yet
#   In order to tie the parameters and have the whole system work, try in order:
#     V 1: manually unlink mul_1 sigmoid params and set them to the mul ones
#       2: make the KernelExpression to_kernel method compose from smaller parts instead of the one-shot conversion
#           (this step also allows addressing the piling-up of variance parameters in products of kernels)
#   Alternatively:
#       0: Make each change class properly wrap the sum of products and mimic each method but without the undesired
#           parameters and the remaining ones shifted up to this class instead of mul

