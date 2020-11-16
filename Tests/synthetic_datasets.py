from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *


# dataset = 'LINx(PER+C)'
# X, Y = generate_data(lambda x: x * np.cos( (x - 5) / 2 )**2, np.linspace(-15, 15, 101), 2, 1)
# correct_k = ProductKE(['LIN'], [SumKE(['PER', 'C'])])._initialise()
# kernel = LIN() * (PER() + C())

# dataset = 'LINxLINxLIN'
# X, Y = generate_data(lambda x: (x + 30) * (x - 5) * (x - 7), np.linspace(-15, 15, 101), 1, 30)
# correct_k = ProductKE(['LIN', 'LIN', 'LIN'])
# kernel = LIN() * LIN() * LIN()

# dataset = 'CP(LIN,PER+C)'
# X, Y = generate_changepoint_data(np.linspace(-20, 30, 101), lambda x: 0.1 * x, lambda x: 2 * np.sin(x), 3, 1, 0.3, True)
# correct_k = ChangeKE('CP', 'LIN', SumKE(['PER', 'C']))
# kernel = CP(LIN(), PER() + C())

# dataset = 'CW(LIN,PER+C)'
# X, Y = generate_changewindow_data(np.linspace(-30, 30, 101), lambda x: 0.1 * x, lambda x: 4 * np.sin(x), -15, 15, 1, 0.3, True)
# correct_k = ChangeKE('CW', 'LIN', SumKE(['PER', 'C']))
# kernel = CW(LIN(), PER() + C())

dataset = 'CW(LIN,LIN*LIN)'
X, Y = generate_changewindow_data(np.linspace(-10, 30, 101), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), 0, 16, 1, 0.3, False)
# X, Y = generate_changewindow_data(np.linspace(-20, 30, 151), lambda x: 0.05 * x, lambda x: 0.05 * x * (x - 14), -10, 16, 1, 0.3, False)
correct_k = ChangeKE('CW', 'LIN', ProductKE(['LIN', 'LIN']))
kernel = CW(LIN(), LIN() * LIN())

# dataset = 'CW(LIN,LIN)'
# X, Y = generate_changewindow_data(np.linspace(-30, 30, 212), lambda x: 0.5 * x, lambda x: 1.5 * x, -15, 15, 1, 0.3, False)
# correct_k = ChangeKE('CW', 'LIN', 'LIN')
# kernel = CW(LIN(), LIN()) # Both non-stationary