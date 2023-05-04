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

from libc.stdlib cimport malloc
cimport cython

#To optimize the C code further, we want to specifcy the type of input arguments
#Example: np.ndarray[DTYPE_t, ndim=3]
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.initializedcheck(False)
def run(double[:,:,:] rays, dict idx):
    cdef int size = len(idx[-1][0])
    cdef void* _buff_addr
    cdef double[:,:] buf
    _buf_addr = malloc(size*3 * sizeof(double*))
    buf = <double[:size,:3]>_buf_addr

    cdef int a = 0
    cdef int b = 0
    for c in range(size):
        a = idx[-1][0][c]
        b = idx[-1][1][c]
        buf[c][0] = rays[0, a, b]
        buf[c][1] = rays[1, a, b]
        buf[c][2] = rays[2, a, b]

    return np.asarray(buf)