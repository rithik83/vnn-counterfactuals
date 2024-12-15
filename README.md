# Neural Network Verification and Counterfactual Explanations

This repository contains tools and scripts for training neural networks, generating counterfactual explanations, and verifying formal robustness guarantees using SAT/SMT solvers.

The project combines **Julia** and **Python** implementations, providing capabilities to:

1. Train and evaluate neural networks.
2. Generate counterfactual points for explainability.
3. Verify robustness properties using formal methods.

---

## Project Structure

```
├── jl_counterfactual_generation
│   ├── counterfactual_points/    # Generated counterfactual results
│   ├── generate_counterfactuals.ipynb  # Jupyter notebook for counterfactual generation
│
├── jl_model_training
│   ├── attacks.jl                # Julia implementation of adversarial attacks
│   ├── evaluate.jl               # Julia implementation to evaluate model robustness
│   ├── model_init_and_train.ipynb  # Jupyter notebook for initializing and training models
│   └── train.jl                  # Julia functionality for adversarial training
│
├── models/                           # Directory to store trained models (note that MedStr: Strong 1 AT)
│
├── py_nn_verification                # Python scripts for neural network verification and result text files
│   ├── clean_res/                    # Results of classically trained network's verification
│   ├── med2_res/                     # Results of Medium 2 AT verification
│   ├── med2_res/, medium_at_res/     # Results of Medium 1 AT verification
│   ├── medstr_res/                   # Results of Strong 1 AT verification
│   ├── strong_at_res/                # Results of Strong AT verification
│   ├── weak_at_res/                  # Results of Weak AT verification
│   └── verify_nn.py                  # Main script for verifying networks
│
├── resources/                        # Additional neural networks used during initial iterations
├── utils/                            # Utility functions (plot)
│
├── .gitignore                        # Git ignore file
├── helloworld.py                     # Python playground script to try out the solver
├── playground.jl                     # Julia playground script to test out ONNX saving and loading
└── README.md                         # Project documentation
```

---

## Dependencies

This project requires **Linux** or **macOS** operating systems. Maraboupy is not supported on Windows.

### Python Dependencies

- `maraboupy`  
- `numpy`

Install the Python dependencies using:

```bash
pip install maraboupy numpy
```

### Julia Dependencies

The following Julia packages are required:

- `ONNXNaiveNASFlux`
- `Flux`
- `CSV`
- `DataFrames`
- `CounterfactualExplanations`

To add these packages, open the Julia REPL and run:

```julia
using Pkg
Pkg.add(["ONNXNaiveNASFlux", "Flux", "CSV", "DataFrames", "CounterfactualExplanations"])
```

---

## Instructions to Run the Code

### 1. Train Neural Networks

Use the `jl_model_training/model_init_and_train.ipynb` notebook for a guided workflow.

### 2. Generate Counterfactual Explanations

To generate counterfactual explanations:

Use the `jl_counterfactual_generation/generate_counterfactuals.ipynb` notebook for a guided workflow

### 3. Verify Neural Network

Verify formal robustness properties using:

```bash
python py_nn_verification/verify_nn.py
```
Here, make sure to select the correct network's file-name (and comment out the rest) and edit the location of the counterfactual points to reflect exactly where the counterfactual points were saved.