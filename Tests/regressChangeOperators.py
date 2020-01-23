import numpy as np

from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.Util.dataAndPlottingUtil import *


X, Y = generate_changepoint_data(np.linspace(-20, 20, 201), lambda x: 0.1 * x, lambda x: 3 * np.sin(x), 0, 1, 0.3, True)
kernel = CP(LIN(), PER() + C())


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 212), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), 0, 15, 1, 0.3, False)
# kernel = CW(LIN(), LIN() * LIN())
# # kernel = CW(LIN() * LIN(), LIN())


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 212), lambda x: 2, lambda x: 2 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
# kernel = CW(C(), PER() + C())
# # kernel = CW(PER() + C(), C())


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 212), lambda x: 0.1 * x, lambda x: 1 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
# kernel = CW(LIN(), PER() + C())
# # kernel = CW(PER + C, LIN)


m = doGPR(X, Y, kernel, 5)
