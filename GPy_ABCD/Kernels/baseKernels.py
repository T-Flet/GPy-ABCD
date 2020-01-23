import GPy.kern as _Gk

from GPy_ABCD.Kernels import linearOffsetKernel as _Lk, changeOperators as _Cs, periodicKernel as _Pk, sigmoidalKernels as _Sk


# WN = _Gk.White(1)
def WN(): return _Gk.White(1)
    # OK

# C = _Gk.Bias(1)
def C(): return _Gk.Bias(1)
    # OK

# LIN = _Gk.Linear(1)
# def LIN(): return _Gk.Linear(1)
    # Not the same as ABCD's; missing horizontal offset; necessary?
# LIN = _Lk.LinearWithOffset(1)
def LIN(): return _Lk.LinearWithOffset(1)
    # The version in ABCD; not sure if a good idea; the horizontal offset is the same as a vertical one, which is just kC

# SE = _Gk.RBF(1)
def SE(): return _Gk.RBF(1)
    # OK

# PER = _Gk.StdPeriodic(1)
# def PER(): return _Gk.StdPeriodic(1)
    # Not the same as ABCD's
# PER = _Pk.PureStdPeriodicKernel(1)
def PER(): return _Pk.PureStdPeriodicKernel(1)
    # OK


# S = _Sk.SigmoidalKernel(1, False)
def S(): return _Sk.SigmoidalKernel(1, False)
# Sr = _Sk.SigmoidalKernel(1, True)
def Sr(): return _Sk.SigmoidalKernel(1, True)


# SIW = _Sk.SigmoidalIndicatorKernelWithWidth(1, False)
def SI(): return _Sk.SigmoidalIndicatorKernelWithWidth(1, False)
# SIWr = _Sk.SigmoidalIndicatorKernelWithWidth(1, True)
def SIr(): return _Sk.SigmoidalIndicatorKernelWithWidth(1, True)

# SI = _Sk.SigmoidalIndicatorKernel(1, False)
def SIT(): return _Sk.SigmoidalIndicatorKernel(1, False)
# SIr = _Sk.SigmoidalIndicatorKernel(1, True)
def SITr(): return _Sk.SigmoidalIndicatorKernel(1, True)

# SIO = _Sk.SigmoidalIndicatorKernelOneLocation(1, False)
def SIO(): return _Sk.SigmoidalIndicatorKernelOneLocation(1, False)
# SIOr = _Sk.SigmoidalIndicatorKernelOneLocation(1, True)
def SIOr(): return _Sk.SigmoidalIndicatorKernelOneLocation(1, True)


CP = _Cs.ChangePointKernel
# CW = _Cs.ChangeWindowKernel
# CW = _Cs.ChangeWindowKernelOneLocation
CW = _Cs.ChangeWindowKernelWithWidth

# CP = _CFs.kCP
# # CW = _CFs.kCW
# # CW = _CFs.kCW2
# CW = _CFs.kCWw
