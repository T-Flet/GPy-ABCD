from matplotlib import pyplot as plt
from Util.util import sampleCurves
from baseKernels import *



kS.plot()
print(kS)
sampleCurves(kS)

kSI.plot()
print(kSI)
sampleCurves(kSI)


plt.show()
