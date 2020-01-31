import numpy as np

from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Util.dataAndPlottingUtil import generate_data
from GPy_ABCD.Kernels.baseKernels import *

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


# X, Y = generate_data(lambda x: np.cos( (x - 5) / 2 )**2, np.linspace(-10, 10, 101), 7, 1)
X, Y = generate_data(lambda x: np.cos( (x - 5) / 2 )**2 + 10, np.linspace(-10, 10, 101), 7, 1)


# doGPR(X, Y, PER(), 10)
doGPR(X, Y, PER() + C(), 10)

# doGPR(X, Y, SE(), 10)
# doGPR(X, Y, SE() + C(), 10)
