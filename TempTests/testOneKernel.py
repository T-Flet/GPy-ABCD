import numpy as np
from matplotlib import pyplot as plt
from Util.util import sampleCurves, doGPR
from baseKernels import *
import GPy
from linearOffsetKernel import LinearWithOffset

#kernel = GPy.kern.LogisticBasisFuncKernel(1, 0) * kLIN
# kernel = kS + kC
# kernel = kPER
kernel = LinearWithOffset(1)
# kernel = kLIN



kernel.plot()
print(kernel)
sampleCurves(kernel)




X = np.linspace(-50, 50, 101)[:, None]

# Y = (1 + np.tanh( -(X - 5) / 10 )) * 5 + np.random.randn(101, 1) #- 100
Y = 20 + 0.5 * X + np.random.randn(101, 1) * 2

doGPR(X, Y, kernel, 10)

