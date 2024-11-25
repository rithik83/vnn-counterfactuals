using Flux, Distances

function FGSM(model, loss, x, y; ϵ = 0.3, clamp_range = (0, 1))
    grads = gradient(x -> loss(model(x), y), x)[1]
    x_adv = clamp.(x + (Float32(ϵ) * sign.(grads)), clamp_range...)
    return x_adv
end

function PGD(model, loss, x, y; ϵ = 0.3, step_size = 0.01, iterations = 40, clamp_range = (0, 1))
    x_adv = clamp.(x + (randn(Float32, size(x)...) * Float32(step_size)), clamp_range...)
    δ = Distances.chebyshev(x, x_adv)
    iteration = 1; while (δ < ϵ) && iteration <= iterations
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iteration += 1
    end
    return x_adv
end