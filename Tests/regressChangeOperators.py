import numpy as np

from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.Util.dataAndPlottingUtil import *


# X, Y = generate_changepoint_data(np.linspace(-20, 20, 201), lambda x: 0.1 * x, lambda x: 3 * np.sin(x), -3, 1, 0.3, True)
# kernel = CP(LIN(), PER() + C())



# X, Y = generate_changewindow_data(np.linspace(-30, 30, 212), lambda x: 0.5 * x, lambda x: 1.5 * x, -15, 15, 1, 0.3, False)
# kernel = CW(LIN(), LIN()) # Both non-stationary


X, Y = generate_changewindow_data(np.linspace(-40, 40, 300), lambda x: 2 * np.sin(x), lambda x: 1.5 * x, -15, 15, 1, 0.3, True)
kernel = CW(PER() + C(), LIN()) # Stationary outside the window
# kernel = CP(PER() + C(), CP(LIN(), PER() + C()))


# X, Y = generate_changewindow_data(np.linspace(-20, 30, 212), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), 0, 16, 1, 0.3, False)
# kernel = CW(LIN(), LIN() * LIN())
# # kernel = CW(LIN() * LIN(), LIN())


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 212), lambda x: 2, lambda x: 2 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
# kernel = CW(C(), PER() + C())
# # kernel = CW(PER() + C(), C())


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 212), lambda x: 0.1 * x, lambda x: 1 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
# kernel = CW(LIN(), PER() + C())
# # kernel = CW(PER + C, LIN)



# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

m = doGPR(X, Y, kernel, 5, optimizer = 'lbfgsb')


# Temporary diagnostics for 3-part changewindow kernel
# print(f'{m.kern.sigmoidal_indicator.location.values} + {m.kern.sigmoidal_indicator.width.values} = {m.kern.sigmoidal_indicator.location.values + m.kern.sigmoidal_indicator.width.values}')
# print(f'{m.kern.sigmoidal.location.values}')
# print(m.kern.sigmoidal.location.values == m.kern.sigmoidal_indicator.location.values + m.kern.sigmoidal_indicator.width.values)
