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
params = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    DECAY = 0.1, # decay (must be between 0 and 1)
    EXT = 0.1*.2, # external stimulation weight
    EXC = 0.1*3, # Excitation gain
    INH = 0.1, # Inhibition gain
    NOISE = 0.0 # noise
)

# Visualize weights
# makie_plotproj(iacn.projections; ncol=3)

# Minimal learning
begin
    iacn = from_def("jets/animals_pools_def", "jets/jets_proj_def", params)
    for proj in iacn.projections
        proj.mat[:, :] .= 10e-5
    end

    reset!(iacn, reset_counter=true)
    state_history = []
    proj_records = []
    ncycle_history = []
    for (i, T) in enumerate([500 for i in 1:6])
        reset!(iacn; reset_counter=false)
        clamp_unit!(iacn, :Civet)
        clamp_pool!(iacn, :Diet)
        states = nsteps!(iacn, T, 0.0001; logstates=true)
        states[!, :iter] .= i
        push!(ncycle_history, (length ∘ unique)(states.t) - 1)
        push!(state_history, states)
        push!(proj_records, iacn.projections |> deepcopy)
        increment_proj!(iacn, :Species => :Civet, :Hidden => :_Civet_, 0.2)
        increment_proj!(iacn, :Diet => :Carnivore, :Hidden => :_Civet_, 0.2)
    end
    state_history = vcat(state_history...)
end

fig = with_theme(Theme(
    fontsize=50,
    Axis = (
        spinewidth=3,
    ),
    Lines = (
        linewidth=10,
    ),
    ScatterLines = (
        linewidth=7,
        markersize=40,
        strokecolor=:white,
        strokewidth=5
    ),
    Vlines = (
        linewidth=8,
    )
)) do
    # Create figure
    fig = Figure(size=(1800, 1600))

    # Plot activations in the Diet pool
    ax = Axis(fig[1, 1], title="Diet pool", ylabel="Activation", yticklabelsvisible=true)
    df = subset(state_history, :pool => ByRow(==(:Diet)))
    for gdf in groupby(mask_first(df, :act), :unit)
        lines!(ax, gdf.t, gdf.act, label=String(gdf.unit[1]), inspector_label=get_label)
    end
    vlines!(ax, cumsum(ncycle_history) .+ 1, color=:gray, linestyle=:dash, alpha=.5, linewidth=5)
    hlines!(ax, [0], color=:gray, linestyle=:dash, alpha=.5, linewidth=5)

    # Plot activations in the :Hidden pool
    ax = Axis(fig[2, 1], title="Hidden pool", ylabel="Activation", yticklabelsvisible=true)
    df = subset(state_history, :pool => ByRow(==(:Hidden)))
    for gdf in groupby(mask_first(df, :act), :unit)
        lines!(ax, gdf.t, gdf.act, label=String(gdf.unit[1]), inspector_label=get_label)
    end
    vlines!(ax, cumsum(ncycle_history) .+ 1, color=:gray, linestyle=:dash, alpha=.5, linewidth=5)
    hlines!(ax, [0], color=:gray, linestyle=:dash, alpha=.5, linewidth=5)

    # Create a container for uncertainty and familiarity vectors and set aside a df with breaks
    ufd = Dict()
    breaks = state_history[:, [:t, :itert, :iter]]

    # Plot uncertainty in the :Diet pool
    ax = Axis(fig[1, 2], ylabel="Final entropy", title="\"Uncertainty\"", yticklabelsvisible=true)
    df = @chain subset(state_history, :pool => ByRow(==(:Diet))) begin
        groupby([:iter, :t])
        combine(:act => uncertainty => :uncertainty)
        @aside ufd["u"] = _[:, :]
        groupby(:iter)
        combine(:uncertainty => last => :finalUncertainty)
    end
    scatterlines!(ax, df.iter, df.finalUncertainty, color=:black)

    # Plot confidence in the :Diet pool (inset)
    ax = Axis(fig[1, 2], ylabel="Final activation", title="\"Confidence\"", width=Relative(0.3), height=Relative(0.3), halign=0.95, valign=0.80, ylabelsize=25, titlesize=28, yticklabelsize=22, xticklabelsize=22, yticklabelsvisible=true)
    df = @chain subset(state_history, :pool => ByRow(==(:Diet))) begin
        groupby([:iter, :pool, :unit])
        combine(:act => last => :confidence)
    end
    for udf in groupby(df, :unit)
        scatterlines!(ax, udf.iter, udf.confidence, label=String(udf.unit[1]), markersize=20, linewidth=5)
    end

    # Plot iter summary of familiarity in the Hidden pool
    ax = Axis(fig[2, 2], ylabel="Final mean activation", title="\"Familiarity\"", yticklabelsvisible=true)
    df = @chain subset(state_history, :pool => ByRow(==(:Hidden))) begin
        groupby([:iter, :t])
        combine(:act => familiarity => :familiarity)
        @aside ufd["f"] = _[:, :]
        groupby(:iter)
        combine(:familiarity => last => :finalFamiliarity)
    end
    scatterlines!(ax, df.iter, df.finalFamiliarity, color=:black)

    # Plot conflict over time for each episode
    ax = Axis(fig[3, 1], ylabel="Familiarity × Uncertainty", title="\"Conflict\"", yticklabelsvisible=true)
    df = @chain innerjoin(ufd["u"], ufd["f"], breaks, on=[:t, :iter]) begin
        transform([:uncertainty, :familiarity] => ((u, f) -> u .* f) => :conflict)
        mask_first(:conflict)
    end
    lines!(ax, df.t, df.conflict, color=:black)
    vlines!(ax, cumsum(ncycle_history) .+ 1, color=:gray, linestyle=:dash, alpha=.5, linewidth=5)

    # Plot conflict readout (Curiosity)
    ax = Axis(fig[3, 2], ylabel="Final conflict", title="\"Curiosity\"", yticklabelsvisible=true)
    df = @chain df begin
        groupby(:iter)
        combine(:conflict => last => :finalConflict)
    end
    scatterlines!(ax, df.iter, df.finalConflict, color=:black)

    fig
end

fig
# DataInspector(fig)

CairoMakie.save("panel-main.svg", fig)