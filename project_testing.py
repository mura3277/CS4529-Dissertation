#Disable numpy array extra logic to clean up function timings
import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

from project_utils import *
from project_geometry import *
from toolbox import *
from analysis_toolbox import *

from dissertation import run_func_profiled, graph_and_log_profiled_solutions, SolutionType

medium = SALT_WATER_1200
device = SV1010_1200Hz

# OBJECTS
# Change object parameters here. Maybe draw them to visualise.
test_tet = Tetrahedron([0, 4, 2], [-2, 4, -1], [2, 4, -1], [0, 3, -1])
test_tri = Triangle([0, 3, -1], [2, 3, -1], [0, 3, 2])
corner_tri = Triangle([0, 1 ,0], [0, 1, 1], [1, 1, 0])

obj_dir = r"C:\Users\Admin\PycharmProjects\Dissertation\OBJ_files"
# human = Mesh(ObjFile(obj_dir + "\human.obj"))
# tyre = Mesh(ObjFile(obj_dir + "\TYRE (1).obj"))

# BACKGROUNDS
# You can use the following backgrounds already imported to this script (see project_geometry.py):
#
# Planes: FLAT_PLANE, FACING_WALL, SLOPING_WALL
# Composites: GAUSSIAN_HILL_X, GAUSSIAN_HOLE_X, ROLLING_BANK_X
# X can take on values 8, 32, 128, 512

# test = Mesh(obj_dir + "/gaussian_hole_8.obj")
test1 = Mesh(GAUSSIAN_HOLE_8)
scene_1 = Scene(background=test1, objects=[])

# Create a scene with the object and background
# scene_tet = Scene(background=FLAT_PLANE, objects=test_tet)

# Create a scene with a .obj model and background
# scene_human = Scene(background=FLAT_PLANE, objects=human)
# scene_tyre = Scene(background=FLAT_PLANE, objects=tyre)

# Run the simulation and save the image. Use your own paths.
log_dir = r"C:\Users\Admin\PycharmProjects\Dissertation\Log_Output"
image_dir = r"C:\Users\Admin\PycharmProjects\Dissertation\Images"

def run_scan():
    with log_terminal_output(log_dir) as log_file:
        print("Scene:", scene_1)
        print("Objects:", scene_1.objects)
        print("Background:", scene_1.background)
        print("Medium:", medium)
        print("Device:", device)
        # Create a scan object and run the simulation
        scan = Scan(A_scan(device, [0, 0, 0], -60, 0, 10, 0.1, "degs", scene=scene_1),"scan", "degs", span=120)
        scan.full_scan(verbosity=2, save_dir=image_dir)


#Iterations for each optimisation step
iterations = 1
#Run profiled simulation for each step
run_func_profiled(run_scan, iterations, [SolutionType.ORIG_FORMAT, SolutionType.ORIG_INTER], graph=False)
# run_func_profiled(run_scan, iterations, [SolutionType.CYTHON_INDEXING, SolutionType.ORIG_INTER], graph=False)

#Generate graph of profiled solutions
graph_and_log_profiled_solutions()