from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.Util.kernelUtil import sampleCurves
from matplotlib import pyplot as plt

## Pure Kernels

WN().plot()
print(WN())
print(sampleCurves(WN()))

C().plot()
print(C())
print(sampleCurves(C()))

LIN().plot()
print(LIN())
print(sampleCurves(LIN()))

SE().plot()
print(SE())
print(sampleCurves(SE()))

PER().plot()
print(PER())
print(sampleCurves(PER()))


plt.show()



## Combinations of kernels

# k_prod = LIN() * PER()
# k_prod.plot()
# print(k_prod)
# sampleCurves(k_prod)
#
# k_add = LIN() + PER()
# k_add.plot()
# print(k_add)
# sampleCurves(k_add)


plt.show()
