import numpy as np

from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *


# X, Y = generate_changewindow_data(np.linspace(-10, 30, 101), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), 0, 16, 1, 0.3, False)
# # X, Y = generate_changewindow_data(np.linspace(-20, 30, 151), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), -10, 16, 1, 0.3, False)
# correct_k = ChangeKE('CW', 'LIN', ProductKE(['LIN'], [SumKE(['LIN', 'C'])]))
# kernel = CW(LIN(), LIN() * (LIN() + C()))
# # correct_k = ChangeKE('CW', 'LIN', ProductKE(['LIN', 'LIN']))
# kernel = CW(LIN(), LIN() * LIN())


# X, Y = generate_changepoint_data(np.linspace(-20, 20, 151), lambda x: 0.1 * x, lambda x: 3 * np.sin(x), -3, 1, 0.3, True)
# correct_k = ChangeKE('CP', 'LIN', SumKE(['PER', 'C']))
# kernel = CP(LIN(), PER() + C())


# X, Y = generate_changewindow_data(np.linspace(-30, 30, 151), lambda x: 0.5 * x, lambda x: 1.5 * x, -15, 15, 1, 0.75, True)
# correct_k = ChangeKE('CW', 'LIN', SumKE(['LIN', 'C']))
# kernel = CW(LIN(), LIN() + C()) # Both non-stationary
# # correct_k = ChangeKE('CW', 'LIN', 'LIN')
# # kernel = CW(LIN(), LIN()) # Both non-stationary


# X, Y = generate_changewindow_data(np.linspace(-40, 40, 151), lambda x: 2 * np.sin(x), lambda x: 1.5 * x, -15, 15, 1, 0.3, True)
# # correct_k = ChangeKE('CW', SumKE(['PER', 'C']), 'LIN')
# # kernel = CW(PER() + C(), LIN()) # Stationary outside the window
# correct_k = ChangeKE('CP', SumKE(['PER', 'C']), ChangeKE('CP', 'LIN', SumKE(['PER', 'C'])))
# kernel = CP(PER() + C(), CP(LIN(), PER() + C()))


# X, Y = generate_changewindow_data(np.linspace(-10, 20, 151), lambda x: 2, lambda x: 2 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
# correct_k = ChangeKE('CW', 'C', SumKE(['PER', 'C']))
# kernel = CW(C(), PER() + C())
# # correct_k = ChangeKE('CW', SumKE(['PER', 'C']), 'C')
# # kernel = CW(PER() + C(), C())


X, Y = generate_changewindow_data(np.linspace(-10, 20, 151), lambda x: 0.1 * x, lambda x: 1 + 3 * np.sin(x*2), 0, 15, 1, 0.3, False)
correct_k = ChangeKE('CW', 'LIN', SumKE(['PER', 'C']))
kernel = CW(LIN(), PER() + C())
# # correct_k = ChangeKE('CW', SumKE(['PER', 'C']), 'LIN')
# # kernel = CW(PER() + C(), C())



# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

# mod = fit_GPy_kern(X, Y, kernel, 10, optimizer ='lbfgsb')

mod = fit_kex(X, Y, correct_k, 10)
model_printout(mod)

plt.show()



# Temporary diagnostics for 3-part changewindow kernel
# print(f'{m.kern.sigmoidal_indicator.location.values} + {m.kern.sigmoidal_indicator.width.values} = {m.kern.sigmoidal_indicator.location.values + m.kern.sigmoidal_indicator.width.values}')
# print(f'{m.kern.sigmoidal.location.values}')
# print(m.kern.sigmoidal.location.values == m.kern.sigmoidal_indicator.location.values + m.kern.sigmoidal_indicator.width.values)


