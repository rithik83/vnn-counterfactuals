from maraboupy import Marabou
import numpy as np
import os

# Loading up our counterfactual data
arr = np.loadtxt("jl_counterfactual_generation/counterfactual_points/medium_at_cfs/data_medium_at.csv", delimiter=",", skiprows=1)
# arr = arr[17:187]

# Loading/setting up our network on Marabou
options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=10)
# filename = 'models/classically_trained.onnx'
# filename = 'models/adv_pgd_strong.onnx'
# filename = 'models/adv_pgd_medstr.onnx'
# filename = 'models/adv_pgd_med2.onnx'
filename = 'models/adv_pgd_medium.onnx'
# filename = 'models/adv_pgd_weak.onnx'

# perturbation bound epsilon and margin for prediction
epsilon = 0.05
margin = -0.00001

sat_instances = 0
unsat_instances = 0
timeouts = 0

for (index, x) in enumerate(arr):

    if index % 1 == 0:
        print("reached datapoint: ", index+1)

    image = x[0:784]
    target = int(x[784])
    print(target)
    
    curr_timeouts = 0
    curr_unsat = 0
    curr_sat = False
    
    for i in range(10):
        if i != target:
            network = Marabou.read_onnx(filename)
            inputVars = network.inputVars[0][0]
            outputVars = network.outputVars[0][0]

            for j in range(len(inputVars)):
                network.setLowerBound(inputVars[j], max(image[j] - epsilon, 0))
                network.setUpperBound(inputVars[j], min(image[j] + epsilon, 1))

            network.addMaxConstraint(set(outputVars), outputVars[i])
            network.addInequality([outputVars[target], outputVars[i]], [1, -1], margin)
            exit_code, vals, stats = network.solve(verbose = False, options = options)

            if exit_code == "sat":
                print("perturbation found!")
                curr_sat = True
                break

            if exit_code == "TIMEOUT":
                print("timeout!!")
                curr_timeouts += 1
            
            if exit_code == "unsat":
                curr_unsat += 1
   
    if curr_sat:
        print("at least one SAT (perturbation exists)")
        sat_instances += 1
    elif curr_timeouts > 0:
        print("TIMEOUT or TIMEOUT/UNSAT (inconclusive)")
        timeouts += 1
    else:
        print("all UNSAT (no perturbation)")
        unsat_instances += 1

print("sat instances (adversarial perturbation found): ", sat_instances)
print("timeouts (inconclusive): ", timeouts)
print("unsat instances (no perturbation found): ", unsat_instances)