from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.baseKernels import *
import numpy as np


# X = np.linspace(-20, 20, 201)[:, None]
# Y = np.concatenate(([0.1 * x for x in X[:100]],
#                     np.array([0])[:, None],
#                     [2 + 3 * np.sin(x) for x in X[101:]])) + np.random.randn(201, 1) * 0.3
#
# kernel = CP(LIN, PER + C)
#
# doGPR(X, Y, kernel, 5)



X = np.linspace(-10, 20, 212)[:, None]

Y = np.concatenate(([0.05 * x for x in X[:70]],
                    np.array([0])[:, None],
                    [0.05 * x * (x - 14) for x in X[71:165]],
                    np.array([0])[:, None],
                    [0.05 * x for x in X[166:]])) + np.random.randn(212, 1) * 0.3
kernel = CW(LIN(), LIN() * LIN())
# kernel = CW(LIN * LIN, LIN)

# Y = np.concatenate(([np.array([2]) for x in X[:70]],
#                     np.array([0])[:, None],
#                     [2 + 3 * np.sin(x*2) for x in X[71:160]],
#                     np.array([0])[:, None],
#                     [np.array([2]) for x in X[161:]])) + np.random.randn(212, 1) * 0.3
# kernel = CW(C(), PER() + C())
# # kernel = CW(PER + C, C)

# Y = np.concatenate(([0.1 * x for x in X[:70]],
#                     np.array([0])[:, None],
#                     [1 + 3 * np.sin(x*2) for x in X[71:160]],
#                     np.array([0])[:, None],
#                     [0.1 * x for x in X[161:]])) + np.random.randn(212, 1) * 0.3
# kernel = CW(LIN(), PER() + C())
# # kernel = CW(PER + C, LIN)


m = doGPR(X, Y, kernel, 5)
