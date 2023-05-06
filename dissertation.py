#Cython Imports
import cython_dtype
import cython_indexing
import cython_views
import cython_raw
import cython_dict

#Imports for execution profiling
import io
import time
import pstats
import cProfile
import re
from pstats import SortKey
import matplotlib.pyplot as plt
import datetime

#Imports for formatting ray array
from enum import Enum, auto
from numpy import array, zeros, arange, where

#Imports for calculating interfunction
from numpy import cross, dot, multiply, NaN

#Keep track of the final profiled data with an list of dictionaries
profiled_runs = []

#Helper Enum class for keeping track and switching between optimisation trategies
class SolutionType(Enum):
    #Solutions: Format Ray Array
    ORIG_FORMAT = auto()
    NO_LIST_COMP = auto()
    INIT_NP_ARRAY = auto()
    CYTHON_DTYPE = auto()
    CYTHON_INDEXING = auto()
    CYTHON_VIEWS = auto()
    CYTHON_RAW = auto()
    CYTHON_DICT = auto()

    #Solutions: Interfunction
    ORIG_INTER = auto()

#Global initial format type
global SOLUTIONS

#Helper for accessing the global SOLUTIONS from other files
def solution_active(solution):
    global SOLUTIONS
    return solution in SOLUTIONS

#Run a python function with profiling output for every passed iteration
def run_func_profiled(func_to_run, iterations, solutions):
    #Setup solution type
    global SOLUTIONS
    SOLUTIONS = solutions

    #Iteration loop
    for i in range(iterations):
        pr = cProfile.Profile()
        pr.enable()
        start = time.time()
        func_to_run()
        end = time.time()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
        ps.print_stats()
        lines = format_profile_output_str(s.getvalue(), end - start)
        profiled_runs.append({"solutions":[s.name for s in solutions], "elapsed": end - start, "lines": lines})
        for l in lines:
            print(l)

def format_profile_output_str(output_str, elapsed):
    lines = output_str.split("\n")[4:25]
    for i in range(len(lines)):
        lines[i] = lines[i][3:]
    lines.insert(0, "Total Elapsed Time: " + str(elapsed))
    return lines

#Optimising formatting the output ray array using an idx lookup.
#This line contributed to over 40 seconds of the 76 second run that was profiled during testing.
def format_ray_array(rays, idx):
    #The original version of the code
    if SolutionType.ORIG_FORMAT in SOLUTIONS:
        return array([(rays[:, idx[-1][0][c], idx[-1][1][c]]) for c in range(len(idx[-1][0]))])
    #Removing the use of list comprehension as this can be a slow operation in Python
    elif SolutionType.NO_LIST_COMP in SOLUTIONS:
        formatted = []
        for c in range(len(idx[-1][0])):
            formatted.append(rays[:, idx[-1][0][c], idx[-1][1][c]])
        numpyArray = array(formatted)
        return numpyArray
    #Initially generating a numpy array with the correct size and shape, then assigning values by index
    elif SolutionType.INIT_NP_ARRAY in SOLUTIONS:
        formatted = zeros((len(idx[-1][0]), 3))
        for c in range(len(idx[-1][0])):
            formatted[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
        return formatted
    #Offload the array formatting to a compiled C binary
    elif SolutionType.CYTHON_DTYPE in SOLUTIONS:
        formatted = cython_dtype.run(rays, idx)
        return formatted
    elif SolutionType.CYTHON_INDEXING in SOLUTIONS:
        formatted = cython_indexing.run(rays, idx)
        return formatted
    elif SolutionType.CYTHON_VIEWS in SOLUTIONS:
        formatted = cython_views.run(rays, idx)
        return formatted
    elif SolutionType.CYTHON_RAW in SOLUTIONS:
        formatted = cython_raw.run(rays, idx)
        return formatted
    elif SolutionType.CYTHON_DICT in SOLUTIONS:
        formatted = cython_dict.run(rays, idx)
        return formatted

def calc_interfunction(rays, pov, p1, v, u):
    rshape = rays.shape[1:]  # Shape of the 2D array of rays
    rays = rays.reshape((3, rays.shape[1] * rays.shape[2])).T  # Reshapes into a 2D array of vectors.
    epsilon = 1e-6
    T = pov - p1  # Vector from p1 to pov (tvec)
    P = cross(rays, v.reshape((1, 3)))  # Cross product of ray and v (pvec)
    S = dot(P, u)  # Dot product of pvec and u (determinant).
    inv_det = where(abs(S) > epsilon, 1 / S, NaN)  # Inverse determinant
    U = multiply(dot(P, T), inv_det)  # Barycentric coordinate u
    # try to whittle down the number of calculations
    if True in (U >= 0) & (U <= 1):  # If u is in the triangle, calculate v and t.
        Q = cross(T, u)  # Cross product of tvec and edge, u. This is constant.
        V = where((U >= 0) & (U <= 1), dot(Q, rays.transpose()), NaN) * inv_det  # Barycentric coordinate v
        t = where(((V >= 0) & (U + V <= 1)), dot(Q, v), NaN) * inv_det  # Distance to intersection point
        t = where(t <= 0, NaN, t)  # If t is negative, the intersection point is behind the pov.
        V = V.reshape(rshape)
        U = U.reshape(rshape)
        t = t.reshape(rshape)
        return U, V, t
    else:
        return None, None, None

def graph_and_log_profiled_solutions():
    plt.rcdefaults()
    fig, ax = plt.subplots()

    run_title = []
    performance = []
    total_time = 0.0
    for r in profiled_runs:
        run_title.append(str(r["solutions"]))
        performance.append(r["elapsed"])
        total_time += float(r["elapsed"])

    y_pos = arange(len(run_title))
    ax.barh(y_pos, performance, align="center")
    ax.set_yticks(y_pos, labels=run_title)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Performance")
    ax.set_title("Execution speed of each profiled optimisation")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig("Results/" + timestamp + ".png")
    plt.show()

    print("Total Time For All Runs: " + str(total_time))