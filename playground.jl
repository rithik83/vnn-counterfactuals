import Pkg
Pkg.add(["Flux", "CounterfactualExplanations", "TaijaData", "LinearAlgebra", "Plots", "ProgressMeter", "Distances", "Statistics", "BSON", "ONNXNaiveNASflux", "NNlib"])

using BSON, NNlib, Flux, ONNXNaiveNASflux

adv_pgd_strong = BSON.load("resources/adv_20ep_32bs_40it_0.01ss_0.3eps.bson")[:adv_pgd_strong]

ONNXNaiveNASflux.save("resources/julia_adv.onnx", adv_pgd_strong)