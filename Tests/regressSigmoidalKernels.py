import numpy as np
from Util.util import doGPR
from baseKernels import *

# np.seterr(all='raise') # Raise exceptions instead of RuntimeWarnings. The exceptions can then be caught by the debugger


X = np.linspace(-50, 50, 101)[:, None]


## Sigmoidal

YS = (1 + np.tanh( -(X - 5) / 10 )) * 5 + np.random.randn(101, 1) * 1 #- 100
doGPR(X, YS, kS + kC, 10)


## Sigmoidal Indicator

# YSI = (    4 * ((1 + np.tanh((X - 5) / 10))/2) * (1 - ((1 + np.tanh((X - 5) / 10))/2))) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# #YSI = (1 - 4 * ((1 + np.tanh((X - 5) / 10))/2) * (1 - ((1 + np.tanh((X - 5) / 10))/2))) * 5 + np.random.randn(101, 1) * 0.5 #- 100
# doGPR(X, YSI, kSI, 10) # + kC
