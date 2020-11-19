

## CORE CONFIGURATION OF BASE KERNELS ##
__INCLUDE_SE_KERNEL = False # The most generic kernel; always a bargain in terms of parameters
__USE_LIN_KERNEL_HORIZONTAL_OFFSET = True # Identifies the polynomial roots; more accurate but one extra parameter per degree
__USE_NON_PURELY_PERIODIC_PER_KERNEL = False # Full standard periodic kernel [MacKay (1998)] instead of only its purely periodic part
__FIX_SIGMOIDAL_KERNELS_SLOPE = True # Hence one parameter fewer for each sigmoidal and related kernel
__USE_INDEPENDENT_SIDES_CHANGEWINDOW_KERNEL = False # Vertical offsets acquired through windows prevent same-instance non-stationary sides kernels from fitting


