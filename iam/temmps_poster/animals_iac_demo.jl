using Pkg; Pkg.activate("IACTools")
using Revise

using IAC
using IACTools
using Chain
using DataFrames
using DelimitedFiles
# using GLMakie
using CairoMakie


# Define network hyperparameters
s = .8
params = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    DECAY = 0.1*s, # decay (must be between 0 and 1)
    EXT = 0.1*.2, # external stimulation weight
    EXC = 0.1*s, # Excitation gain
    INH = 0.1*.7, # Inhibition gain
    NOISE = 0.01 # noise
)

# Visualize weights
# makie_plotproj(iacn.projections; ncol=3)

# Demo
begin
    iacn = from_def("jets/animals_pools_def", "jets/jets_proj_def", params)
    reset!(iacn, reset_counter=true)
    state_history = []
    for (i, T) in enumerate([80])
        reset!(iacn; reset_counter=false)
        # clamp_pool!(iacn, :Species)
        # clamp_unit!(iacn, :Mammal)
        # clamp_unit!(iacn, :Aquatic)
        # clamp_unit!(iacn, :Herbivore)
        # clamp_unit!(iacn, :Americas)
        clamp_unit!(iacn, :Coyote)
        states = nsteps!(iacn, T; logstates=true)
        push!(state_history, states)
    end
    state_history = vcat(state_history...)
end

# Plot all activations
fig = with_theme(Theme(
    fontsize=30,
    linewidth=4,
    Axis = (
        spinewidth=2,
    )
)) do
    # plot_acts(state_history; ncols=2, limits=(nothing, (params.MIN, params.MAX)), select=[:Species, :Hidden])
    plot_acts(state_history; ncols=4, limits=(nothing, (params.MIN, params.MAX)), select=[:Class, :Continent, :Activity, :Diet])
end

save("poster-panel01.svg", fig)

@chain state_history begin
    sort([:pool, :t, :act], rev=true)
    subset(:t => ByRow(==(30)))
    groupby(:pool)
    @aside for g in _ display(first(g, 5)) end
end
