#Cython Imports
import cython_main

#Imports for execution profiling
import io
import pstats
import cProfile
import re
from pstats import SortKey

#Imports for formatting ray array
from numpy import array

#Run a python function with profiling output for every iteration
def run_func_profiled(func_to_run, iterations):
    #Remove loop for a single run or set to 1.
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
#Oringial: array([(rays[:, idx[-1][0][c], idx[-1][1][c]]) for c in range(len(idx[-1][0]))])
def format_ray_array(rays, idx):
    return array([(rays[:, idx[-1][0][c], idx[-1][1][c]]) for c in range(len(idx[-1][0]))])


print(cython_main.test(5))