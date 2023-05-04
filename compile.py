import subprocess
from subprocess import check_output

subprocess.Popen("cd C:/Users/Admin/PycharmProjects/Dissertation", shell=True)
print(check_output(["python", "cython_build.py", "build_ext", "--inplace"], shell=True))
