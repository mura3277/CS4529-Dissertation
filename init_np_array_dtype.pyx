#Import the numpy module
import numpy as np

#Import the Cython specific numpy module with added compile time information
cimport numpy as np

#As per Cython docs: It's necessary to call "import_array" if you use any part of the numpy API
np.import_array()

#Datatype for the numpy arrays. np.single for single-precision floating point values
DTYPE = np.float32

#Assigns a corresponding compile-time type to DTYPE_t
ctypedef np.float32_t DTYPE_t

def run(np.ndarray rays, dict idx):
    cdef np.ndarray formatted = np.zeros((len(idx[-1][0]), 3), dtype=DTYPE)
    for c in range(len(idx[-1][0])):
        formatted[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
    return formatted