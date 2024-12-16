using Pkg; Pkg.activate("Familiarity")
using CairoMakie
using DataFrames
using Distributions
using LinearAlgebra
using Random

import LogExpFunctions: softmax, logistic
import InvertedIndices: Not
import StatsBase: sample

function Δa(act::Vector{T}, net::Vector{T}; rest::T=-0.1, mx::T=1.0, mn::T=-0.2, γ::T=0.1) where T <: AbstractFloat
    Δ = Vector{Float64}(undef, length(net))
    for i in 1:length(Δ)
        if net[i] >= 0
            Δ[i] = net[i] * (mx - act[i])
        else
            Δ[i] = net[i] * (act[i] - mn)
        end
    end
    Δ .-= γ * (act .- rest)
    return Δ
end

function Δw(W::Matrix{T}, enact::Vector{T}, exact::Vector{T}; λ::T) where T <: AbstractFloat
    aᵢaⱼ = enact * exact'
    adj = zero(aᵢaⱼ) .+ 1.0
    adj[aᵢaⱼ .> 0] .= -1.0
    Δ = λ .* aᵢaⱼ .* (1 .+ W .* adj)
    return Δ
end

function oja(W::Matrix{T}, enact::Vector{T}, exact::Vector{T}; λ::T) where T <: AbstractFloat
    return λ .* enact * (exact - W' * enact)'
end

begin
    # Random.seed!(1)
    
    MAX = 1.0
    MIN = 0.0
    REST = 0.0
    γ = 0.5 # decay (must be between 0 and 1)
    ε = 1.0

    # α = 0.5
    λ = .05
    σ = 1e-5
    T = 500

    # Define phases and events
    label_phases = [UnitRange(i, i+20) for i in 10:40:T]#[10:20, 100:150]#[2:3]#, 120:130, 220:T]
    binding_phases = []#[120:130]
    hebbian_phases = label_phases
    reset_steps = [i+21 for i in 10:40:T]

    # Construct network
    nhidden = 10
    nlabels = 10
    nfeatures = 10

    pools = (
        label = collect(1:nlabels),
        color = collect(1:nfeatures),
        shape = collect(1:nfeatures),
        hidden = collect(1:nhidden)
    )
    poolkeys = keys(pools)

    # Initialize vectors for network's values
    ext = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    net = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    pact = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    act = NamedTuple{poolkeys}(map(pool -> fill(REST, length(pool)), pools))

    # Initialize weight matrices
    W = NamedTuple{poolkeys}(map(pool -> rand(Uniform(0.00, 0.001), nhidden, length(pool)), pools))
    M = NamedTuple{poolkeys}(map(pool -> diagm(0 => pool |> zero) .- 0.001, pools))

    # Create data containers
    df = DataFrame(step=Int[], pool=String[], unit=Int[], net=Float64[], pact=Float64[], act=Float64[])
    wstore =  NamedTuple{poolkeys}([[] for k in poolkeys])


    # Simulate n steps
    for step in 1:T
        # Clamp peripheral units familiarization
        if any([step in sp for sp in label_phases])
            ext.label[1] = MAX
        end

        # Clamp peripheral units for pairing
        if any([step in pp for pp in binding_phases])
            ext.label[1] = MAX
            ext.color[1] = MAX
            ext.shape[1] = MAX
            ext.orien[1] = MAX
        end

        # Update hidden units
        net.hidden[:] = +([W[k] * act[k] for k in poolkeys]...) + M.hidden * act.hidden + ε .* ext.hidden
        # net.hidden[:] += rand(Normal(0, σ), length(net.hidden))
        pact.hidden[:] += Δa(pact.hidden, net.hidden; rest=REST, mx=MAX, mn=MIN, γ=γ)
        # pact.hidden[:] = min.(pact.hidden, MAX)
        # pact.hidden[:] = max.(pact.hidden, MIN)
        act.hidden[:] = rand(nhidden) .<= pact.hidden .|> Float64

        # # Update peripheral units
        for k in poolkeys[1:end-1]
            net[k][:] = W[k]' * act.hidden + M[k] * act[k] + ε .* ext[k]
            # net[k][:] += rand(Normal(0, σ), length(net[k]))
            pact[k][:] += Δa(pact[k], net[k]; rest=REST, mx=MAX, mn=MIN, γ=γ)
            # pact[k][:] = min.(pact[k], MAX)
            # pact[k][:] = max.(pact[k], MIN)
            act[k][:] = (rand(nfeatures) .<= pact[k]) .|> Float64
        end

        # Hebbian learning
        if any([step in hp for hp in hebbian_phases])
            for k in poolkeys[1:end-1]
                # W[k][:, :] += Δw(W[k], act.hidden, act[k]; λ=λ)
                W[k][:, :] += oja(W[k], act.hidden, act[k]; λ=λ)
            end
        end

        # Reset network
        if step in reset_steps
            for k in poolkeys
                ext[k][:] .= 0.0
                # net[k][:] .= 0.0
                act[k][:] .= REST
            end
        end

        # Save data
        for k in poolkeys
            for (i, (n, p, a)) in enumerate(zip(net[k], pact[k], act[k]))
                push!(df, [step, String(k), i, n, p, a])
            end
            if k != :hidden
                push!(wstore[k], vec(W[k][:, :]))
            end
        end
    end
end


begin 
    # Visualize
    fig = Figure(size=(1400, 1000))
    for (i, k) in poolkeys |> enumerate
        ax1 = Axis(fig[i, 1], ylabel="net($k)",
            limits=(
                (1, T),
                (nothing, nothing)
            )
        )
        ax2 = Axis(fig[i, 2], ylabel="pact($k)",
            limits=(
                (1, T),
                (-0.05, 1.05)
            )
        )
        ax3 = Axis(fig[i, 3], ylabel="weights ($k)",
            limits=(
                (1, T),
                (nothing, nothing)
            )
        )
        ax4 = Axis(fig[i, 4], ylabel="active($k)",
            limits=(
                (1, T),
                (nothing, nothing)
            )
        )
        ax5 = Axis(fig[i, 5], ylabel="Conflict($k)",
            limits=(
                (1, T),
                (nothing, nothing)
            )
        )
        subdf = subset(df, :pool => ==(String(k)) |> ByRow)
        for (ii, gdf) in groupby(subdf, :unit) |> enumerate
            lines!(ax1, 1:T, gdf.net, alpha=.4, linewidth=2)
            lines!(ax2, 1:T, gdf.pact, alpha=.4, linewidth=2)
            active = gdf.act .> 0
            scatter!(ax4, collect(1:T)[active], gdf.act[active] .+ ii .- 1, color=:black, marker=:vline)
        end
        if k != "hidden"
            for row in eachrow(hcat(wstore[k]...))
                lines!(ax3, 1:T, row, alpha=.7, linewidth=1)
            end
        end
        conflict = combine(groupby(subdf, [:step, :pool]), [:net, :pact] => ((net, prob) -> mean(net) * sum(softmax(prob) .* log.(1 ./ softmax(prob)))) => :conflict).conflict
        # conflict = combine(groupby(subdf, [:step, :pool]), [:net, :act] => ((net, prob) -> mean(net) * sum(prob .* log.(1 ./ prob))) => :conflict).conflict
        lines!(ax5, 1:T, conflict, color=:black, linewidth=2)
    end
    fig
end 

# save("res2.png", fig)