#Cython Imports
import cython_dtype
import cython_indexing
import cython_views
import cython_raw
import cython_dict

#Imports for execution profiling
import io
import pstats
import cProfile
import re
from pstats import SortKey

#Imports for formatting ray array
from enum import Enum, auto
from numpy import array, zeros

#Helper Enum class for keeping track and switching between optimisation trategies
class RayFormat(Enum):
    ORIGINAL = auto()
    NO_LIST_COMP = auto()
    INIT_NP_ARRAY = auto()
    CYTHON_DTYPE = auto()
    CYTHON_INDEXING = auto()
    CYTHON_VIEWS = auto()
    CYTHON_RAW = auto()
    CYTHON_DICT = auto()

#Global initial format type
global FORMAT_TYPE

#Run a python function with profiling output for every iteration
def run_func_profiled(func_to_run, iterations, format_type):
    #Setup solution type
    global FORMAT_TYPE
    FORMAT_TYPE = format_type

    #Iteration loop
    for i in range(iterations):
        pr = cProfile.Profile()
        pr.enable()
        func_to_run()
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        format_profile_output_str(s.getvalue())

def format_profile_output_str(output_str):
    lines = output_str.split("\n")[4:40]
    for l in lines:
        print(l)

#Optimising formatting the output ray array using an idx lookup.
#This line contributed to over 40 seconds of the 76 second run that was profiled during testing.
def format_ray_array(rays, idx):
    #The original version of the code
    if FORMAT_TYPE.value == RayFormat.ORIGINAL.value:
        return array([(rays[:, idx[-1][0][c], idx[-1][1][c]]) for c in range(len(idx[-1][0]))])
    #Removing the use of list comprehension as this can be a slow operation in Python
    elif FORMAT_TYPE.value == RayFormat.NO_LIST_COMP.value:
        formatted = []
        for c in range(len(idx[-1][0])):
            formatted.append(rays[:, idx[-1][0][c], idx[-1][1][c]])
        numpyArray = array(formatted)
        return numpyArray
    #Initially generating a numpy array with the correct size and shape, then assigning values by index
    elif FORMAT_TYPE.value == RayFormat.INIT_NP_ARRAY.value:
        formatted = zeros((len(idx[-1][0]), 3))
        for c in range(len(idx[-1][0])):
            formatted[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
        return formatted
    #Offload the array formatting to a compiled C binary
    elif FORMAT_TYPE.value == RayFormat.CYTHON_DTYPE.value:
        formatted = cython_dtype.run(rays, idx)
        return formatted
    elif FORMAT_TYPE.value == RayFormat.CYTHON_INDEXING.value:
        formatted = cython_indexing.run(rays, idx)
        return formatted
    elif FORMAT_TYPE.value == RayFormat.CYTHON_VIEWS.value:
        formatted = cython_views.run(rays, idx)
        return formatted
    elif FORMAT_TYPE.value == RayFormat.CYTHON_RAW.value:
        formatted = cython_raw.run(rays, idx)
        return formatted
    elif FORMAT_TYPE.value == RayFormat.CYTHON_DICT.value:
        formatted = cython_dict.run(rays, idx)
        return formatted