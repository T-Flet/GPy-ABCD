import GPy.kern as _Gk

from GPy_ABCD.Kernels import linearKernel as _Lk, linearOffsetKernel as _LOk, changeOperators as _Cs, periodicKernel as _Pk, sigmoidalKernels as _Sk



#### CORE CONFIGURATION OF BASE KERNELS ####
__INCLUDE_SE_KERNEL = False
__USE_LIN_KERNEL_HORIZONTAL_OFFSET = False
__USE_NON_PURELY_PERIODIC_PER_KERNEL = False
# __USE_SIGMOIDAL_KERNELS_SLOPE = True
#### CORE CONFIGURATION OF BASE KERNELS ####



## Useful Kernel Sets

base_kerns = frozenset(['WN', 'C', 'LIN', 'SE', 'PER']) if __INCLUDE_SE_KERNEL else frozenset(['WN', 'C', 'LIN', 'PER'])

stationary_kerns = frozenset(['WN', 'C', 'SE', 'PER'])
# addition_idempotent_kerns = frozenset(['WN', 'C'])
# multiplication_idempotent_kerns = frozenset(['WN', 'C', 'SE'])
# multiplication_zero_kerns = frozenset(['WN']) # UNLESS LIN!!!!!!! I.E. ZERO ONLY FOR STATIONARY KERNELS
# multiplication_identity_kerns = frozenset(['C'])

base_order = {'PER': 1, 'WN': 2, 'SE': 3, 'C': 4, 'LIN': 5}
    # Then sort by: sorted(LIST, key=lambda SYM: baseOrder[SYM])

base_sigmoids = frozenset(['S', 'Sr', 'SI', 'SIr'])


## Base Kernels

# WN = _Gk.White(1)
def WN(): return _Gk.White(1)

# C = _Gk.Bias(1)
def C(): return _Gk.Bias(1)

# LIN = _Gk.Linear(1)
# def LIN(): return _Gk.Linear(1) # Not the same as ABCD's; missing horizontal offset
# LIN = _Lk.Linear(1)
def LIN(): return _Lk.Linear(1) # Not the same as ABCD's; missing horizontal offset
if __USE_LIN_KERNEL_HORIZONTAL_OFFSET: # The version in ABCD; not sure if a good idea; the horizontal offset is the same as a vertical one, which is just kC
    # LIN = _Lk.LinearWithOffset(1)
    def LIN(): return _LOk.LinearWithOffset(1)

# SE = _Gk.RBF(1)
def SE(): return _Gk.RBF(1)

# PER = _Pk.PureStdPeriodicKernel(1)
def PER(): return _Pk.PureStdPeriodicKernel(1)
if __USE_NON_PURELY_PERIODIC_PER_KERNEL: # Not the same as ABCD's
    # PER = _Gk.StdPeriodic(1)
    def PER(): return _Gk.StdPeriodic(1)


## Sigmoidal Kernels

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


# Change-Operator Kernels

CP = _Cs.ChangePointKernel
# CW = _Cs.ChangeWindowKernel
# CW = _Cs.ChangeWindowKernelOneLocation
CW = _Cs.ChangeWindowKernelWithWidth

# CP = _CFs.kCP
# # CW = _CFs.kCW
# # CW = _CFs.kCW2
# CW = _CFs.kCWw
