import numpy as np

from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *

from GPy_ABCD.Kernels.linearKernel import Linear
from GPy_ABCD.Kernels.linearOffsetKernel import LinearWithOffset
from GPy_ABCD.Kernels.baseKernels import C, __USE_LIN_KERNEL_HORIZONTAL_OFFSET


# X, Y = generate_data(lambda x: 2 * x + 20, np.linspace(-20, 20, 201), 1, 0.5)
# # kernel = Linear(1)
# # kernel = Linear(1) + C()
# kernel = LinearWithOffset(1)


# X, Y = generate_data(lambda x: - (x - 2) * (x + 3), np.linspace(-20, 20, 201), 1, 20)
# # kernel = Linear(1) * Linear(1)
# # kernel = Linear(1) * Linear(1) + C()
# # kernel = LinearWithOffset(1) * LinearWithOffset(1)
# # kernel = LinearWithOffset(1) * LinearWithOffset(1) + C()
# kernel = ProductKE(['LIN', 'LIN']).to_kernel()
# # kernel = SumKE(['C'], [ProductKE(['LIN', 'LIN'])]).to_kernel()


X, Y = generate_data(lambda x: - (x - 12) * (x + 18) * (x - 7), np.linspace(-20, 20, 201), 1, 20)
kernel = ProductKE(['LIN', 'LIN', 'LIN']).to_kernel()


# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

mod = fit_GPy_kern(X, Y, kernel, 20, optimizer = GPy_optimisers[0])

# mod = fit_kex(X, Y, correct_k, 10)
# model_printout(mod)

plt.show()

    # VERDICT: LinearWithOffset is just better: better fits (even accounting for extra params) which is interpretable (the offsets are just roots)
    #           And having a single variance per product is even better


