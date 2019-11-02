from sigmoidalKernels import SigmoidalKernel, SigmoidalIndicatorKernel


def kCP(k1, k2):
    """
    :param k1: Kernel before the changepoint
    :param k2: Kernel after the changepoint
    :return: A kernel composed of k1 being replaced by k2 at some changepoint
    """
    return k1 * SigmoidalKernel(1, True) + k2 * SigmoidalKernel(1, False)


def kCW(k1, k2):
    """
    :param k1: Kernel before and after the window
    :param k2: Kernel during the window
    :return: A kernel of k1 replaced by k2 in some changewindow
    """
    return k1 * SigmoidalIndicatorKernel(1, True) + k2 * SigmoidalIndicatorKernel(1, False)






