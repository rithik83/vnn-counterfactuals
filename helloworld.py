from maraboupy import Marabou
import numpy as np
from torchvision import datasets, transforms

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()

print(train_set_array[0:1].flatten().shape)

options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=10)

filename = 'resources/julia_clean.onnx'
# filename = 'resources/julia_adv.onnx'
network = Marabou.read_onnx(filename)

inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

# print("inputVars shape: ", inputVars.shape)
# print("outputVars shape: ", outputVars.shape)
# print("outputVars: ", outputVars)

epsilon = 0.1
index = 4523
image = train_set_array[index: index + 1].flatten() / 255

correct_class = train_set.targets.numpy()[index]
print("correct class: ", correct_class)


for i in range(len(inputVars)):
    network.setLowerBound(inputVars[i], max(image[i] - epsilon, 0))
    network.setUpperBound(inputVars[i], min(image[i] + epsilon, 1))

margin = -0.0001

for i in range(len(outputVars)):
  print("i: ", i)
  if i != correct_class:
    network.addMaxConstraint(set(outputVars), outputVars[i])
    network.addInequality([outputVars[correct_class], outputVars[i]], [1, -1], margin)
    exit_code, vals, stats = network.solve(verbose = False, options = options)

    print("satisfiability? ", exit_code)

    # if solution found, break
    if len(vals) > 0:
      for j, var in enumerate(outputVars):
        print(f"output {j}: {vals[var]}")
      print(f"maxclass: {i}")
      inputPoint = np.zeros((1, 784))
      for j, var in enumerate(inputVars):
        inputPoint[0][j] = vals[j]
      
      meval = network.evaluateWithMarabou(inputPoint, options = options)[0]
      onnxeval = network.evaluateWithoutMarabou(inputPoint)[0]

      print("marabou eval: ", meval)
      print("onnx eval: ", onnxeval)
      
      break










# print("\nConvolutional Network with Max Pool Example")
# filename = 'resources/classic_cnn.onnx'
# network = Marabou.read_onnx(filename)

# # # %%
# # Get the input and output variable numbers; [0] since first dimension is batch size
# inputVars = network.inputVars[0]
# outputVars = network.outputVars[0]

# # %% 
# # Test Marabou equations against onnxruntime at an example input point
# inputPoint = np.zeros(inputVars.shape)
# print("inputvars shape: ", inputVars.shape)
# marabouEval = network.evaluateWithMarabou([inputPoint], options = options)[0]
# onnxEval = network.evaluateWithoutMarabou([inputPoint])[0]

# # # %%
# # The two evaluations should produce the same result
# print("Marabou Evaluation:")
# print(marabouEval)
# print("\nONNX Evaluation:")
# print(onnxEval)
# print("\nDifference:")
# print(onnxEval - marabouEval)
# assert max(abs(onnxEval - marabouEval).flatten()) < 1e-3


# print("\nConvolutional Network Example")
# filename = 'resources/KJ_TinyTaxiNet.onnx'
# network = Marabou.read_onnx(filename)

# # %%
# # Get the input and output variable numbers; [0] since first dimension is batch size
# inputVars = network.inputVars[0][0]
# outputVars = network.outputVars[0][0]

# # %%
# # Setup a local robustness query
# delta = 0.03
# for h in range(inputVars.shape[0]):
#     for w in range(inputVars.shape[1]):
#         network.setLowerBound(inputVars[h][w][0], 0.5-delta)
#         network.setUpperBound(inputVars[h][w][0], 0.5+delta)

# # %%
# # Set output bounds
# network.setLowerBound(outputVars[0], 6.0)

# # %%
# # Call to Marabou solver (should be SAT)
# print("Check query with less restrictive output constraint (Should be SAT)")
# exitCode, vals, stats = network.solve(options = options)
# assert( exitCode == "sat")
# assert len(vals) > 0

# print("vals: ", vals)

# # %%
# # Set more restrictive output bounds
# network.setLowerBound(outputVars[0], 6.55)

# # %%
# # Call to Marabou solver (should be UNSAT)
# print("Check query with more restrictive output constraint (Should be UNSAT)")
# exitCode, vals, stats = network.solve(options = options)
# assert( exitCode == "unsat")
# assert len(vals) == 0