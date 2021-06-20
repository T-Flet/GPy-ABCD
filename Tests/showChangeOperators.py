from GPy_ABCD.Util.kernelUtil import sampleCurves
from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.Kernels import sigmoidalKernels as _Sk
from matplotlib import pyplot as plt


## Direct Sigmoidal effects on a given kernel

kBase = C()

# # Ascending Sigmoid
# k_asc = kBase * _Sk.SigmoidalKernel(1, True)
# k_asc.plot()
# print(k_asc)
# print(sampleCurves(k_asc))
#
# # Descending Sigmoid
# k_desc = kBase * SigmoidalKernel(1, False)
# k_desc.plot()
# print(k_desc)
# print(sampleCurves(k_desc))
#
# # Peak Sigmoid
# k_peak = kBase * SigmoidalIndicatorKernel(1, True)
# k_peak.plot()
# print(k_peak)
# print(sampleCurves(k_peak))
#
# # Hole Sigmoid
# k_hole = kBase * SigmoidalIndicatorKernel(1, False)
# k_hole.plot()
# print(k_hole)
# print(sampleCurves(k_hole))


plt.show()



## Change Operators

k_CP = CP(LIN(), PER())
k_CP.plot()
print(k_CP)
print(sampleCurves(k_CP, xlims = (-3., 5.)))

k_CW = CW(LIN(), PER())
k_CW.plot()
print(k_CW)
print(sampleCurves(k_CW))


plt.show()
