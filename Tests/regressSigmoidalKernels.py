from GPy_ABCD.Util.kernelUtil import doGPR
from GPy_ABCD.Kernels.baseKernels import *
import numpy as np

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger

def tanhSigmoid(x):
    return (1 + np.tanh(x)) / 2
def tanhSigTwoLocIndicatorHeight(location, stop_location, slope):
    return 2 * tanhSigmoid(((stop_location - location) / 2) / slope) - 1


X = np.linspace(-50, 50, 101)[:, None]


## Sigmoidal

# YS = tanhSigmoid((X - 5) / 10) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# doGPR(X, YS, S, 5)
# doGPR(X, YS, Sr, 5)
# doGPR(X, YS, S + C, 5)
# YS = -YS
# doGPR(X, YS, S, 5)
# doGPR(X, YS, Sr, 5)
# doGPR(X, YS, S + C, 5)
# YS = tanhSigmoid((X - 5) / (-10)) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# doGPR(X, YS, S, 5)
# doGPR(X, YS, Sr, 5)
# doGPR(X, YS, S + C, 5)
# YS = -YS
# doGPR(X, YS, S, 5)
# doGPR(X, YS, Sr, 5)
# doGPR(X, YS, S + C, 5)

# I.e: S only fits functions moving from 0 to +ve or -ve values; Sr ones going TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint


## Sigmoidal Indicator with Width

# YSI = ((tanhSigmoid((X + 8) / 10) + tanhSigmoid((X - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10)) * 5 + np.random.randn(101, 1) * 0.4 #- 100
# doGPR(X, YSI, SI, 5)
# doGPR(X, YSI, SIr, 5)
# doGPR(X, YSI, SI + C, 5)
# YSI = -YSI
# doGPR(X, YSI, SI, 5)
# doGPR(X, YSI, SIr, 5)
# doGPR(X, YSI, SI + C, 5)
# YSI = (1 - ((tanhSigmoid((X + 8) / 10) + tanhSigmoid((X - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10))) * 5 + np.random.randn(101, 1) * 0.4 #- 100
# doGPR(X, YSI, SI, 5)
# doGPR(X, YSI, SIr, 5)
# doGPR(X, YSI, SI + C, 5)
# YSI = -YSI
# doGPR(X, YSI, SI, 5)
# doGPR(X, YSI, SIr, 5)
# doGPR(X, YSI, SI + C, 5)

# I.e: Same concept as for S, SIT and SIO etc.: SI only fits functions moving from 0 to +ve or -ve values and then going back to 0; SIr ones going temporarily TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal peak/well shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint


## Sigmoidal Indicator With One Location

# YSI = (4 * tanhSigmoid((X - 5) / 10) * (1 - tanhSigmoid((X - 5) / 10))) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# print(doGPR(X, YSI, SIO, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIOr, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIO + C, 5).kern.sigmoidal_indicator.start_and_end_locations())
# YSI = -YSI
# print(doGPR(X, YSI, SIO, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIOr, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIO + C, 5).kern.sigmoidal_indicator.start_and_end_locations())
# YSI = (1 - 4 * tanhSigmoid((X - 5) / 10) * (1 - tanhSigmoid((X - 5) / 10))) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# print(doGPR(X, YSI, SIO, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIOr, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIO + C, 5).kern.sigmoidal_indicator.start_and_end_locations())
# YSI = -YSI
# print(doGPR(X, YSI, SIO, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIOr, 5).kern.start_and_end_locations())
# print(doGPR(X, YSI, SIO + C, 5).kern.sigmoidal_indicator.start_and_end_locations())

# I.e: Same concept as for S etc.: SIO only fits functions moving from 0 to +ve or -ve values and then going back to 0; SIOr ones going temporarily TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal peak/well shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint


## Sigmoidal Indicator with Two Locations

# YSI = ((tanhSigmoid((X + 8) / 10) + tanhSigmoid((X - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10)) * 5 + np.random.randn(101, 1) * 0.4 #- 100
# doGPR(X, YSI, SIT, 5)
# doGPR(X, YSI, SITr, 5)
# doGPR(X, YSI, SIT + C, 5)
# YSI = -YSI
# doGPR(X, YSI, SIT, 5)
# doGPR(X, YSI, SITr, 5)
# doGPR(X, YSI, SIT + C, 5)
# YSI = (1 - ((tanhSigmoid((X + 8) / 10) + tanhSigmoid((X - 5) / (-10)) - 1) / tanhSigTwoLocIndicatorHeight(-8, 5, 10))) * 5 + np.random.randn(101, 1) * 0.4 #- 100
# doGPR(X, YSI, SIT, 5)
# doGPR(X, YSI, SITr, 5)
# doGPR(X, YSI, SIT + C, 5)
# YSI = -YSI
# doGPR(X, YSI, SIT, 5)
# doGPR(X, YSI, SITr, 5)
# doGPR(X, YSI, SIT + C, 5)

# I.e: Same concept as for S and SIO etc.: SIT only fits functions moving from 0 to +ve or -ve values and then going back to 0; SITr ones going temporarily TO 0
#   Also, adding a constant makes either version fit any vaguely sigmoidal peak/well shape
#   NOTE: this hinges on self.slope HAVING the +ve constraint
