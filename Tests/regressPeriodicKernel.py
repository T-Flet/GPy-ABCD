import numpy as np

from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *


# X, Y = generate_data(lambda x: np.cos( (x - 5) / 2 ), np.linspace(-10, 10, 101), 7, 1)
X, Y = generate_data(lambda x: np.cos( (x - 5) / 2 )**2 + 10, np.linspace(-10, 10, 101), 7, 1)


# kernel = PER()
# correct_k = SumKE(['PER'])
kernel = PER() + C()
correct_k = SumKE(['PER', 'C'])
# kernel = SE()
# correct_k = SumKE(['SE'])
# kernel = SE() + C()
# correct_k = SumKE(['SE', 'C'])



# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

# mod = fit_GPy_kern(X, Y, kernel, 10, optimizer ='lbfgsb')

mod = fit_kex(X, Y, correct_k, 10)


mod.change_plotting_library()
mod.plot()[0].show()


model_printout(mod)

plt.show()


