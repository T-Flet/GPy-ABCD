import GPy.kern as _Gk
from Kernels import sigmoidalKernels as _Sk, periodicKernel as _Pk, linearOffsetKernel as _Lk, changeOperators as _Cs

WN = _Gk.White(1) # OK
C = _Gk.Bias(1) # OK
# LIN = _Gk.Linear(1) # Not the same as ABCD's; missing horizontal offset; necessary?
LIN = _Lk.LinearWithOffset(1) # The version in ABCD; not sure if a good idea; the horizontal offset is the same as a vertical one, which is just kC
SE = _Gk.RBF(1) # OK
# PER = _Gk.StdPeriodic(1) # Not the same as ABCD's
PER = _Pk.PureStdPeriodicKernel(1)


S = _Sk.SigmoidalKernel(1, False)
Sr = _Sk.SigmoidalKernel(1, True)
SI = _Sk.SigmoidalIndicatorKernel(1, False)
SIr = _Sk.SigmoidalIndicatorKernel(1, True)


CP = _Cs.kCP
CW = _Cs.kCW