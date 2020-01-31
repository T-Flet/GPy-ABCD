import numpy as np

from GPy_ABCD.Kernels.linearKernel import Linear
from GPy_ABCD.Kernels.linearOffsetKernel import LinearWithOffset
from GPy_ABCD.Kernels.baseKernels import C, __USE_LIN_KERNEL_HORIZONTAL_OFFSET
from GPy_ABCD.KernelExpansion.kernelExpression import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.kernelUtil import doGPR, score_ps, BIC, AIC, AICc
from GPy_ABCD.Models.modelSearch import fit_model_list_parallel


# X, Y = generate_data(lambda x: 2 * x + 20, np.linspace(-20, 20, 201), 1, 0.5)
# # kernel = Linear(1)
# # kernel = Linear(1) + C()
# kernel = LinearWithOffset(1)


X, Y = generate_data(lambda x: - (x - 2) * (x + 3), np.linspace(-20, 20, 201), 1, 20)
# kernel = Linear(1) * Linear(1)
# kernel = Linear(1) * Linear(1) + C()
# kernel = ProductKE(['LIN', 'LIN']).to_kernel()
kernel = SumKE(['C'], [ProductKE(['LIN', 'LIN'])]).to_kernel()
# kernel = LinearWithOffset(1) * LinearWithOffset(1)
# kernel = LinearWithOffset(1) * LinearWithOffset(1) + C()


m = doGPR(X, Y, kernel, 5, BIC)

    # VERDICT: LinearWithOffset is just better: better fits (even accounting for extra params) which is interpretable (the offsets are just roots)
