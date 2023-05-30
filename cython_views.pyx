#Import the numpy module
import numpy as np

#Import the Cython specific numpy module with added compile time information
cimport numpy as np

#As per Cython docs: It's necessary to call "import_array" if you use any part of the numpy API
np.import_array()

#Datatype for the numpy arrays. np.single for single-precision floating point values
DTYPE = np.double

#Assigns a corresponding compile-time type to DTYPE_t
ctypedef np.double_t DTYPE_t

cimport cython

#To optimize the C code further, we want to specifcy the type of input arguments
#Example: np.ndarray[DTYPE_t, ndim=3]
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.initializedcheck(False)
def run(double[:,:,:] rays, dict idx):
    #Initialise the numpy array at the correct size
    cdef double[:,:] formatted = np.empty((len(idx[-1][0]), 3), dtype=DTYPE)

    #Integer cache variables
    cdef int a = 0
    cdef int b = 0

    #Sub-Dimensional array cache
    cdef double[:] outer_dimn = formatted[0]

    #Main loop
    for c in range(len(idx[-1][0])):
        #Cache dictionary access
        a = idx[-1][0][c]
        b = idx[-1][1][c]
        outer_dimn = formatted[c]

        #Assign rays to buffer
        outer_dimn[0] = rays[0, a, b]
        outer_dimn[1] = rays[1, a, b]
        outer_dimn[2] = rays[2, a, b]

    #Output formatted array
    return formatted