from matplotlib import pyplot as plt
from Util.util import sampleCurves
from Kernels.baseKernels import *



S.plot()
print(S)
sampleCurves(S)

Sr.plot()
print(Sr)
sampleCurves(Sr)


SI.plot()
print(SI)
sampleCurves(SI)

SIr.plot()
print(SIr)
sampleCurves(SIr)


SIT.plot()
print(SIT)
sampleCurves(SIT)

SITr.plot()
print(SITr)
sampleCurves(SITr)



plt.show()
