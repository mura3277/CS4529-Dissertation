#Import the numpy zeros function
import numpy as np

def run(rays, idx):
    formatted = np.zeros((len(idx[-1][0]), 3))
    for c in range(len(idx[-1][0])):
        formatted[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
    return formatted