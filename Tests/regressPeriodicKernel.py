from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.baseKernels import *
import numpy as np

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


X = np.linspace(-10, 10, 101)[:, None]

Y = np.cos( (X - 5) / 2 )**2 * 7 + np.random.randn(101, 1) * 1 #- 100

doGPR(X, Y, PER + C, 10)


