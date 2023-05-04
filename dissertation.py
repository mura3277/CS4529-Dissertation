#Cython Imports
import cython_main

#Imports for execution profiling
import io
import pstats
import cProfile
import re
from pstats import SortKey

#Imports for formatting ray array
from enum import Enum
from numpy import array, zeros

#Helper Enum class for keeping track and switching between optimisation trategies
class RayFormat(Enum):
    ORIGINAL = 1
    NO_LIST_COMP = 2
    INIT_NP_ARRAY = 3
#Global initial format type
global FORMAT_TYPE

#Run a python function with profiling output for every iteration
def run_func_profiled(func_to_run, iterations, format_type):
    global FORMAT_TYPE
    FORMAT_TYPE = format_type
    #Iteration loop
    for i in range(iterations):
        pr = cProfile.Profile()
        pr.enable()
        func_to_run()
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

#Optimising formatting the output ray array using an idx lookup.
#This line contributed to over 40 seconds of the 76 second run that was profiled during testing.
def format_ray_array(rays, idx):
    #The original version of the code
    if FORMAT_TYPE.value == RayFormat.ORIGINAL.value:
        return array([(rays[:, idx[-1][0][c], idx[-1][1][c]]) for c in range(len(idx[-1][0]))])
    #Removing the use of list comprehension as this can be a slow operation in Python
    elif FORMAT_TYPE.value == RayFormat.NO_LIST_COMP.value:
        new = []
        for c in range(len(idx[-1][0])):
            new.append(rays[:, idx[-1][0][c], idx[-1][1][c]])
        return array(new)
    #Initially generating a numpy array with the correct size and shape, then assigning values by index
    elif FORMAT_TYPE.value == RayFormat.INIT_NP_ARRAY.value:
        new = zeros((len(idx[-1][0]), 3))
        for c in range(len(idx[-1][0])):
            new[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
        return new

def run_cython_tests():
    print(cython_main.test(5))