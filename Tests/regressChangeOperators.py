import numpy as np
from Util.util import doGPR
from Kernels.baseKernels import *


X = np.linspace(-10, 10, 101)[:, None]

Y = np.concatenate(([0.1 * x for x in X[:50]],
                    np.array([0])[:, None],
                    [2 + 3 * np.sin(x) for x in X[51:]])) + np.random.randn(101, 1) * 0.3

kernel = CP(LIN, PER + C)

doGPR(X, Y, kernel, 5)
