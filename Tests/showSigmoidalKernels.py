from matplotlib import pyplot as plt
from Util.kernelUtil import sampleCurves
from Kernels.baseKernels import *



S.plot()
print(S)
sampleCurves(S)

Sr.plot()
print(Sr)
sampleCurves(Sr)


SIT.plot()
print(SIT)
sampleCurves(SIT)

SITr.plot()
print(SITr)
sampleCurves(SITr)


SIO.plot()
print(SIO)
sampleCurves(SIO)

SIOr.plot()
print(SIOr)
sampleCurves(SIOr)


SI.plot()
print(SI)
sampleCurves(SI)

SIr.plot()
print(SIr)
sampleCurves(SIr)



plt.show()
