# Confidence-curiosity phenomenon
using Pkg; Pkg.activate("IACTools")
using Revise

using AlgebraOfGraphics
using IAC
using IACTools
using Chain
using DataFrames, DataFramesMeta
using DelimitedFiles
using Distributions
using GLMakie


df, iacn = let
    # Define network hyperparameters
    netparams = (
        MAX = 1.0,
        MIN = -0.2,
        REST = -0.1,
        DECAY = 0.1, # Decay (must be between 0 and 1)
        EXT = 0.4, # External stimulation weight
        EXC = 0.1, # Excitation gain
        INH = 0.15, # Inhibition gain
        NOISE = 0.0, # Noise
        DMG = Beta(1, 100), # Truncated(Exponential(.01), 0.0, 1.0)
    )

    tol = 0.0001
    iacn = from_csv_table("iam/def/hp_table.csv", netparams; hidden_ids=:id)
    for proj in iacn.projections
        damage = rand(netparams.DMG, size(proj.mat)...)
        proj.mat[:, :] = max.(proj.mat .- damage, 0.0)
    end
    reset!(iacn, reset_counter=true)
    state_history = []
    for (i, T) in enumerate([500])
        reset!(iacn; reset_counter=false)
        clamp_unit!(iacn, "potter")
        states = nsteps!(iacn, T; logstates=true)
        push!(state_history, states)
    end
    state_history = vcat(state_history...)
    state_history, iacn
end

# Plot activations
fig = let
    fig = Figure()
    layers = acts_plot_layers(df)
    lims = (nothing, (iacn.params.MIN, iacn.params.MAX))
    draw!(fig, layers, axis=(; limits=lims))
    fig
end