#Cython Imports
import cython_main

#Imports for execution profiling
import cProfile
import re
from pstats import SortKey

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


print(cython_main.test(5))