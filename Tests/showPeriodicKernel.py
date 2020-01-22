from GPy_ABCD.Util.kernelUtil import sampleCurves
from GPy_ABCD.Kernels.baseKernels import *
from matplotlib import pyplot as plt



PER.plot()
print(PER)
sampleCurves(PER)

plt.show()
