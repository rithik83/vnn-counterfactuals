import numpy as np
import os

print(os.getcwd())

data = np.loadtxt("jl_counterfactual_generation/data.csv", delimiter=',', skiprows=1)

print("size of data: ", data.shape)
print("data first pt: ", data[0])