import os, subprocess
import matplotlib.pyplot as plt
from subprocess import check_output
import datetime

dir_root = "C:/Users/Admin/PycharmProjects/Dissertation/"

def process_obj_files():
    #Process all the obj files in the "OBJ_files" directory
    for f in os.listdir(dir_root + 'OBJ_files'):
        if os.path.isfile(dir_root + 'OBJ_files_optimised/' + f):
            continue
        #Execute a subprocess to run the following commands to compile the obj files using go and the obj-simplify project
        subprocess.Popen("cd " + dir_root, shell=True)
        check_output(["go", "run", "./obj-simplify-master", "-in", "../OBJ_files/" + f, "-out", "../OBJ_files_optimised/" + f, "-no-duplicates"], shell=True)

#Compute a list of the file sizes of all models, before and after as well as their names
def get_obj_sizes():
    before = []
    after = []
    names = []
    for f in os.listdir(dir_root + 'OBJ_files'):
        before.append(os.path.getsize(dir_root + 'OBJ_files/' + f) / 1024) #Divide by 1024 to convert from bytes to kilobytes
        after.append(os.path.getsize(dir_root + 'OBJ_files_optimised/' + f) / 1024) #Divide by 1024 to convert from bytes to kilobytes
        names.append(f)
    return before, after, names

#Compute the difference in file size
def compute_diff(before, after):
    diff = []
    for i in range(len(before)):
        if after[i] < before[i]:
            diff.append((abs(after[i] - before[i]) / before[i]) * 100.0)
        else:
            diff.append(0)
    return diff

#Generate the graphs for the computed data
def graph_sizes(before, after, diff, names):
    # Graph showing the difference in sizes between the old and new models
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.barh(range(len(before)), before, align='center')
    plt.barh(range(len(after)), after, align='center')
    plt.title("File size result of each optimised model")
    plt.xlabel("Size (kilobytes)")
    plt.ylabel("Model")
    plt.savefig("Results/diff-" + timestamp + ".png")
    plt.clf()

    #Graph showing the sizes of the optimised files
    plt.barh(range(len(diff)), diff, align='center')
    plt.title("Difference in file size of each model")
    plt.xlabel("Amount (kilobytes)")
    plt.ylabel("Model")
    plt.savefig("Results/size-" + timestamp + ".png")

#Process model files
process_obj_files()
#Get sizes
before, after, names = get_obj_sizes()
#Compute difference
diff = compute_diff(before, after)
#Generate graphs
graph_sizes(before, after, diff, names)
