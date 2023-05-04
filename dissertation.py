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

#Imports for formatting ray array
from enum import Enum, auto
from numpy import array, zeros, arange

#Keep track of the final profiled data with an list of dictionaries
profiled_runs = []

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
        start = time.time()
        func_to_run()
        end = time.time()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
        ps.print_stats()
        lines = format_profile_output_str(s.getvalue(), end - start)
        profiled_runs.append({"name":format_type.name, "elapsed": end - start, "lines": lines})
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

def graph_profiled_outputs():
    plt.rcdefaults()
    fig, ax = plt.subplots()

    formats = []
    performance = []
    for r in profiled_runs:
        formats.append(r["name"])
        performance.append(r["elapsed"])

    y_pos = arange(len(formats))
    ax.barh(y_pos, performance, align="center")
    ax.set_yticks(y_pos, labels=formats)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Performance")
    ax.set_title("Execution speed of each profiled optimisation")

    plt.savefig("results.png")
    plt.show()