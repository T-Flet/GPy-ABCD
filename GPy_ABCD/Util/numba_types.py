'''This file is an old, reduced, Python-3.8-compatible version of the one in https://github.com/T-Flet/Python-Generic-Util'''
from numba import njit, vectorize, guvectorize, stencil, typeof, b1, f8, i8
from numba.core.types import Array
from numba.core.types.containers import Tuple

from typing import Union


## This script serves to export common numba functions and type signatures easily and without namespace pollution
__all__ = ['njit', 'typeof',
           'b1', 'f8', 'i8', # The basic types used in the project
           'b1A', 'f8A', 'i8A', # Shorthand for their C-consecutive 1d arrays
           'b1A2', 'f8A2', 'i8A2', # Shorthand for their C-consecutive 2d arrays
           'nTup', 'Union'] # numba's Tuple (renamed to nTup to avoid clashes) and typing's Union


b1A = b1[::1]
f8A = f8[::1]
i8A = i8[::1]

b1A2 = b1[:, ::1]
f8A2 = f8[:, ::1]
i8A2 = i8[:, ::1]

def nTup(*args): return Tuple(args)


