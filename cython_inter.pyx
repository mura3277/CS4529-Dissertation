#Import the numpy module
import numpy as np

#Import the Cython specific numpy module with added compile time information
cimport numpy as np

#As per Cython docs: It's necessary to call "import_array" if you use any part of the numpy API
np.import_array()

from libc.stdlib cimport malloc
cimport cython

#To optimize the C code further, we want to specifcy the type of input arguments
#Example: np.ndarray[DTYPE_t, ndim=3]
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def run2(np.ndarray[np.double_t, ndim=3] rays, np.ndarray[np.int_t, ndim=1] pov, np.ndarray[np.double_t, ndim=1] p1, np.ndarray[np.double_t, ndim=1] v, np.ndarray[np.double_t, ndim=1] u):
    cdef int rshapeX = rays.shape[1]
    cdef int rshapeZ = rays.shape[2]

    #rshape = rays.shape[1:]  # Shape of the 2D array of rays
    rshape = (rshapeX, rshapeZ)

    #rays = rays.reshape((3, rays.shape[1] * rays.shape[2])).T # Reshapes into a 2D array of vectors.
    #cdef double[:,:] rays2D = <double[:rshapeX, :rshapeZ]>rays

    cdef np.ndarray[np.double_t, ndim=2] rays2D = rays.reshape((3, rays.shape[1] * rays.shape[2])).T

    cdef double epsilon = 1e-6
    cdef np.ndarray[np.double_t] T = pov - p1  # Vector from p1 to pov (tvec)
    cdef np.ndarray[np.double_t, ndim=2] P = np.cross(rays2D, v.reshape((1, 3)))  # Cross product of ray and v (pvec)
    cdef np.ndarray[np.double_t] S = np.dot(P, u)  # Dot product of pvec and u (determinant).
    cdef np.ndarray[np.double_t] S_abs = np.abs(S)
    cdef np.ndarray[np.double_t] one_over_S = np.divide(1)
    cdef np.ndarray[np.double_t] inv_det = np.where(S_abs > epsilon, 1 / S, np.NaN)  # Inverse determinant


    cdef np.ndarray[np.double_t] U = np.multiply(np.dot(P, T), inv_det)  # Barycentric coordinate u

    # try to whittle down the number of calculations
    if True in (U >= 0) & (U <= 1):  # If u is in the triangle, calculate v and t.
        Q = np.cross(T, u)  # Cross product of tvec and edge, u. This is constant.
        V = np.where((U >= 0) & (U <= 1), np.dot(Q, rays2D.transpose()), np.NaN) * inv_det  # Barycentric coordinate v
        t = np.where(((V >= 0) & (U + V <= 1)), np.dot(Q, v), np.NaN) * inv_det  # Distance to intersection point
        t = np.where(t <= 0, np.NaN, t)  # If t is negative, the intersection point is behind the pov.
        V = V.reshape(rshape)

        #U = U.reshape(rshape)

        t = t.reshape(rshape)
        return U.reshape(rshape), V, t
    else:
        return None, None, None

def run(rays, pov, p1, v, u):
    rshape = rays.shape[1:]  # Shape of the 2D array of rays
    rays = rays.reshape((3, rays.shape[1] * rays.shape[2])).T

    epsilon = 1e-6
    T = pov - p1  # Vector from p1 to pov (tvec)
    P = np.cross(rays, v.reshape((1, 3)))  # Cross product of ray and v (pvec)
    S = np.dot(P, u)  # Dot product of pvec and u (determinant).
    inv_det = np.where(abs(S) > epsilon, 1 / S, np.NaN)  # Inverse determinant
    U = np.multiply(np.dot(P, T), inv_det)  # Barycentric coordinate u

    # try to whittle down the number of calculations
    if True in (U >= 0) & (U <= 1):  # If u is in the triangle, calculate v and t.
        Q = np.cross(T, u)  # Cross product of tvec and edge, u. This is constant.
        V = np.where((U >= 0) & (U <= 1), np.dot(Q, rays.transpose()), np.NaN) * inv_det  # Barycentric coordinate v
        t = np.where(((V >= 0) & (U + V <= 1)), np.dot(Q, v), np.NaN) * inv_det  # Distance to intersection point
        t = np.where(t <= 0, np.NaN, t)  # If t is negative, the intersection point is behind the pov.
        V = V.reshape(rshape)
        U = U.reshape(rshape)
        t = t.reshape(rshape)
        return U, V, t
    else:
        return None, None, None