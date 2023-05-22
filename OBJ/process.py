import os, subprocess
import matplotlib.pyplot as plt
from subprocess import check_output
import datetime

dir_root = "C:/Users/Admin/PycharmProjects/Dissertation/"

def process_obj_files():
    for f in os.listdir(dir_root + 'OBJ_files'):
        if os.path.isfile(dir_root + 'OBJ_files_optimised/' + f):
            continue
        subprocess.Popen("cd " + dir_root, shell=True)
        check_output(["go", "run", "./obj-simplify-master", "-in", "../OBJ_files/" + f, "-out", "../OBJ_files_optimised/" + f, "-no-duplicates"], shell=True)

def get_obj_sizes():
    before = []
    after = []
    for f in os.listdir(dir_root + 'OBJ_files'):
        before.append(os.path.getsize(dir_root + 'OBJ_files/' + f))
        after.append(os.path.getsize(dir_root + 'OBJ_files_optimised/' + f))
    return before, after

def compute_diff(before, after):
    diff = []
    for i in range(len(before)):
        if after[i] < before[i]:
            diff.append((abs(after[i] - before[i]) / before[i]) * 100.0)
        else:
            diff.append(0)
    return diff

def graph_sizes(before, after, diff):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.bar(range(len(before)), before, align='center')
    plt.bar(range(len(after)), after, align='center')
    plt.savefig("Results/diff-" + timestamp + ".png")
    plt.clf()

    plt.bar(range(len(diff)), diff, align='center')
    plt.savefig("Results/size-" + timestamp + ".png")

process_obj_files()
before, after = get_obj_sizes()
diff = compute_diff(before, after)
graph_sizes(before, after, diff)
