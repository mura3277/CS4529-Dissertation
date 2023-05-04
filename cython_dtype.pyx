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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#To optimize the C code further, we want to specifcy the type of input arguments
#Example: np.ndarray[DTYPE_t, ndim=3]
def run(np.ndarray[DTYPE_t, ndim=3] rays, dict idx):
    cdef np.ndarray[DTYPE_t, ndim=2] formatted = np.empty((len(idx[-1][0]), 3), dtype=DTYPE)
    for c in range(len(idx[-1][0])):
        formatted[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
    return formatted