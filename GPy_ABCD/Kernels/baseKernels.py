import GPy.kern as _Gk

from GPy_ABCD.Kernels import linearKernel as _Lk, linearOffsetKernel as _LOk, changeOperators as _Cs, periodicKernel as _Pk, sigmoidalKernels as _Sk
from GPy_ABCD.Kernels import changeWindowThreePart as _CWTk, changeWindowShiftedSides as _CWSk



#### CORE CONFIGURATION OF BASE KERNELS ####
__INCLUDE_SE_KERNEL = False # The most generic kernel; always a bargain in terms of parameters
__USE_LIN_KERNEL_HORIZONTAL_OFFSET = True # Identifies the polynomial roots; more accurate but one extra parameter per degree
__USE_NON_PURELY_PERIODIC_PER_KERNEL = False # Full standard periodic kernel [MacKay (1998)] instead of only its purely periodic part
__FIX_SIGMOIDAL_KERNELS_SLOPE = True # Hence one parameter fewer for each sigmoidal and related kernel
__USE_INDEPENDENT_SIDES_CHANGEWINDOW_KERNEL = False # Vertical offsets acquired through windows prevent same-instance non-stationary sides kernels from fitting
#### CORE CONFIGURATION OF BASE KERNELS ####



## Useful Kernel Sets

base_kerns = frozenset(['WN', 'C', 'LIN', 'SE', 'PER']) #if __INCLUDE_SE_KERNEL else frozenset(['WN', 'C', 'LIN', 'PER'])
base_sigmoids = frozenset(['S', 'Sr', 'SI', 'SIr'])

stationary_kerns = frozenset(['WN', 'C', 'SE', 'PER'])
non_stationary_kerns = base_sigmoids.union(base_kerns - stationary_kerns)
# addition_idempotent_kerns = frozenset(['WN', 'C'])
# multiplication_idempotent_kerns = frozenset(['WN', 'C', 'SE'])
# multiplication_zero_kerns = frozenset(['WN']) # UNLESS LIN!!!!!!! I.E. ZERO ONLY FOR STATIONARY KERNELS
# multiplication_identity_kerns = frozenset(['C'])

base_order = {'PER': 1, 'WN': 2, 'SE': 3, 'C': 4, 'LIN': 5}
    # Then sort by: sorted(LIST, key=lambda SYM: baseOrder[SYM])



## Base Kernels

def WN(): return _Gk.White(1)

def C(): return _Gk.Bias(1)

# def LIN(): return _Gk.Linear(1) # Not the same as ABCD's; missing horizontal offset
def LIN(): return _Lk.Linear(1) # Not the same as ABCD's; missing horizontal offset
if __USE_LIN_KERNEL_HORIZONTAL_OFFSET: # The version in ABCD; not sure if a good idea; the horizontal offset is the same as a vertical one, which is just kC
    def LIN(): return _LOk.LinearWithOffset(1)

def SE(): return _Gk.RBF(1)

def PER(): return _Pk.PureStdPeriodicKernel(1)
if __USE_NON_PURELY_PERIODIC_PER_KERNEL: # Not the same as ABCD's
    def PER(): return _Gk.StdPeriodic(1)


## Sigmoidal Kernels

def S(): return _Sk.SigmoidalKernel(1, False, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def Sr(): return _Sk.SigmoidalKernel(1, True, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)


def SI(): return _Sk.SigmoidalIndicatorKernel(1, False, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def SIr(): return _Sk.SigmoidalIndicatorKernel(1, True, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)

def SICW(): return _Sk.SigmoidalIndicatorKernelCentreWidth(1, False, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def SICWr(): return _Sk.SigmoidalIndicatorKernelCentreWidth(1, True, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)

def SIT(): return _Sk.SigmoidalIndicatorKernelTwoLocations(1, False, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def SITr(): return _Sk.SigmoidalIndicatorKernelTwoLocations(1, True, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)

def SIO(): return _Sk.SigmoidalIndicatorKernelOneLocation(1, False, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def SIOr(): return _Sk.SigmoidalIndicatorKernelOneLocation(1, True, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)


# Change-Operator Kernels

def CP(first, second): return _Cs.ChangePointKernel(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
def CW(first, second): return _Cs.ChangeWindowKernel(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
# def CW(first, second): return _Cs.ChangeWindowKernelCorrectedWidth(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
# def CW(first, second): return _Cs.ChangeWindowKernelAlternating(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)

# def CW(first, second): return _Cs.ChangeWindowKernelCentreWidth(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
# def CW(first, second): return _Cs.ChangeWindowKernelTwoLocations(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
# def CW(first, second): return _Cs.ChangeWindowKernelOneLocation(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)

if __USE_INDEPENDENT_SIDES_CHANGEWINDOW_KERNEL:
    def CW(first, second): return _CWSk.ChangeWindowKernelShiftedSides(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)
    # def CW(first, second): return _CWTk.ChangeWindowKernelIndependent(first, second, fixed_slope = __FIX_SIGMOIDAL_KERNELS_SLOPE)


# CP = _CFs.kCP
# # CW = _CFs.kCW
# # CW = _CFs.kCW2
# CW = _CFs.kCWw
