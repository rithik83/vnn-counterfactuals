# Neural Network Verification and Counterfactual Explanations

### 1. Training Neural Networks

The `jl_model_training/model_init_and_train.ipynb` notebook has the workflow for training networks. Networks are saved in ONNX format in `models_new`

### 2. Generate Counterfactual Explanations

To generate counterfactual explanations:

The `jl_counterfactual_generation/generate_counterfactuals.ipynb`contains a workflow to generate counterfactual explanations, saved in `jl_counterfactual_generation/counterfactual_points_new`

### 3. Verify Neural Network

Neural network verification is done via alpha-beta-CROWN, a state-of-the-art verifier. The `config.yaml` file consists of the configuration we used to verify the network