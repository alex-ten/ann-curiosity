using Pkg; Pkg.activate("Familiarity")
using CairoMakie
using Distributions
using Random

import LogExpFunctions: softmax, logistic
import StatsBase: sample

ecdf(x) = 1 - exp(-x)

begin
    Random.seed!(123)
    
    α = 2.0
    τ = 1.0

    nhidden = 4
    nlabels = 3
    nfeatures = 3

    hues = Makie.wong_colors()[1:nhidden]

    labels = collect(1:nlabels)
    colors = collect(1:nfeatures)
    shapes = collect(1:nfeatures)
    oriens = collect(1:nfeatures)

    # Define vectors for network's net input values
    net = (
        label = nlabels |> zeros,
        color = nfeatures |> zeros,
        shape = nfeatures |> zeros,
        orien = nfeatures |> zeros,
        hidden = nhidden |> zeros
    )

    # Define vectors for network's activation values
    act = (
        label = nlabels |> zeros,
        color = nfeatures |> zeros,
        shape = nfeatures |> zeros,
        orien = nfeatures |> zeros,
        hidden = nhidden |> zeros
    )

    # Define weights to the hidden 
    W = (
        label = rand(Uniform(0.00, 0.05), nhidden, nlabels),
        color = rand(Uniform(0.00, 0.05), nhidden, nfeatures),
        shape = rand(Uniform(0.00, 0.05), nhidden, nfeatures),
        orien = rand(Uniform(0.00, 0.05), nhidden, nfeatures)
    )

    # Define biases to word units
    bias = Dict(
        :word => [7, 5]
    )
    
    # Process information for n steps
    fig = Figure()
    ax = Axis(fig[1, 1], limits=(nothing, (0, 1)))

    for step in 1:20
        # Clamp peripheral units (external input)
        act.label[1] = 1.0

        # Update net input to the hidden units
        net.hidden[:] = +(
            W.label * act.label,
            W.color * act.color,
            W.shape * act.shape
        )

        # Update activation of the hidden units
        # act.hidden[:] = ecdf.(net.hidden)
        act.hidden[:] = softmax(net.hidden / τ)
        
        # Update net input to peripheral units
        net.label[:] = W.label' * act.hidden
        net.color[:] = W.color' * act.hidden
        net.shape[:] = W.shape' * act.hidden
        net.orien[:] = W.orien' * act.hidden

        # Update net input to peripheral units
        act.label[:] = ecdf.(net.label)
        act.color[:] = ecdf.(net.color)
        act.shape[:] = ecdf.(net.shape)
        act.orien[:] = ecdf.(net.orien)

        # Hebbian learning
        W.label[:] = W.label + α .* (act.hidden * act.label')

        scatter!(ax, fill(step, nhidden), act.hidden, color=hues)
        # scatter!(ax, [step], [acts[:word][2]], color=:red)
    end
    fig
end