import numpy as np

from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *


def sig(x): return (1 + x / (1 + abs(x))) / 2
def sigTwoLocIndicatorHeight(location, stop_location, slope):
    arg = (stop_location - location) / (2 * slope)
    return arg / (1 + abs(arg))


X = np.linspace(-50, 50, 101)[:, None]


## Sigmoidal

# YS = sig((X - 5) / 10) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# fit_GPy_kern(X, YS, S(), 5)
# fit_GPy_kern(X, YS, Sr(), 5)
# fit_GPy_kern(X, YS, S() + C(), 5)
# YS = -YS
# fit_GPy_kern(X, YS, S(), 5)
# fit_GPy_kern(X, YS, Sr(), 5)
# fit_GPy_kern(X, YS, S() + C(), 5)
# YS = sig((X - 5) / (-10)) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# fit_GPy_kern(X, YS, S(), 5)
# fit_GPy_kern(X, YS, Sr(), 5)
# fit_GPy_kern(X, YS, S() + C(), 5)
# YS = -YS
# fit_GPy_kern(X, YS, S(), 5)
# fit_GPy_kern(X, YS, Sr(), 5)
# fit_GPy_kern(X, YS, S() + C(), 5)

# I.e: S only fits functions moving from 0 to +ve or -ve values; Sr ones going TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint


## Sigmoidal Indicator with Start Location and Width

YSI = ((sig((X + 8) / 10) + sig((X - 5) / (-10)) - 1) / sigTwoLocIndicatorHeight(-8, 5, 10)) * 5 + np.random.randn(101, 1) * 0.4 #- 100
fit_GPy_kern(X, YSI, SI(), 5)
fit_GPy_kern(X, YSI, SIr(), 5)
fit_GPy_kern(X, YSI, SI() + C(), 5)
YSI = -YSI
fit_GPy_kern(X, YSI, SI(), 5)
fit_GPy_kern(X, YSI, SIr(), 5)
fit_GPy_kern(X, YSI, SI() + C(), 5)
YSI = (1 - ((sig((X + 8) / 10) + sig((X - 5) / (-10)) - 1) / sigTwoLocIndicatorHeight(-8, 5, 10))) * 5 + np.random.randn(101, 1) * 0.4 #- 100
fit_GPy_kern(X, YSI, SI(), 5)
fit_GPy_kern(X, YSI, SIr(), 5)
fit_GPy_kern(X, YSI, SI() + C(), 5)
YSI = -YSI
fit_GPy_kern(X, YSI, SI(), 5)
fit_GPy_kern(X, YSI, SIr(), 5)
fit_GPy_kern(X, YSI, SI() + C(), 5)

# I.e: Same concept as for S, SIT and SIO etc.: SI only fits functions moving from 0 to +ve or -ve values and then going back to 0; SIr ones going temporarily TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal peak/well shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint


