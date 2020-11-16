from GPy_ABCD.Util.kernelUtil import sampleCurves
from GPy_ABCD.Kernels.baseKernels import *
from matplotlib import pyplot as plt



S().plot()
print(S())
sampleCurves(S())

Sr().plot()
print(Sr())
sampleCurves(Sr())


SI().plot()
print(SI())
sampleCurves(SI())

SIr().plot()
print(SIr())
sampleCurves(SIr())



plt.show()
