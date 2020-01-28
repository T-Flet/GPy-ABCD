import numpy as np

from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.linearKernel import Linear
from GPy_ABCD.Kernels.linearOffsetKernel import LinearWithOffset
from GPy_ABCD.Kernels.baseKernels import C
from GPy_ABCD.Util.dataAndPlottingUtil import *


X, Y = generate_data(lambda x: 2 * x + 20, np.linspace(-20, 20, 201), 1, 0.5)
# kernel = Linear(1)
kernel = Linear(1) + C()
# kernel = LinearWithOffset(1)


m = doGPR(X, Y, kernel, 5)
