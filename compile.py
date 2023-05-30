import subprocess
from subprocess import check_output

#Run a terminal command from within python
subprocess.Popen("cd C:/Users/Admin/PycharmProjects/Dissertation", shell=True)
#Run the Cython build script
print(check_output(["python", "cython_build.py", "build_ext", "--inplace"], shell=True))
