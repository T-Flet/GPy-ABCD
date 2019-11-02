import GPy.kern as _Gk
from changeOperators import kCP, kCW # Here for export purposes
import sigmoidalKernels as _Sk
import periodicKernel as _Pk
import linearOffsetKernel as _Lk


kWN = _Gk.White(1) # OK
kC = _Gk.Bias(1) # OK
# kLIN = _Gk.Linear(1) # Not the same as ABCD's; missing horizontal offset; necessary?
kLIN = _Lk.LinearWithOffset(1) # The version in ABCD; not sure if a good idea; the horizontal offset is the same as a vertical one, which is just kC
kSE = _Gk.RBF(1) # OK
# kPER = _Gk.StdPeriodic(1) # Not the same as ABCD's, but that lengthy implementation can be done later
kPER = _Pk.PureStdPeriodicKernel(1)


kS = _Sk.SigmoidalKernel(1, False)
kSr = _Sk.SigmoidalKernel(1, True)
kSI = _Sk.SigmoidalIndicatorKernel(1, False)
kSIr = _Sk.SigmoidalIndicatorKernel(1, True)