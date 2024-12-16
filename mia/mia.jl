using Pkg; Pkg.activate("Familiarity")
using CairoMakie
using DataFrames
using Distributions
using InvertedIndices: Not
using LinearAlgebra
using LogExpFunctions: softmax, logistic
using Random
using StatsBase: sample, Weights


function oh(size::Int, i::Int)
    v = zeros(Int, size)
    v[i] = 1
    return v
end

function oja(W::Matrix{T}, enact::Vector{T}, exact::Vector{T}; λ::T) where T <: AbstractFloat
    return λ .* enact * (exact - W' * enact)'
end

function inhmat(val::Float64, size::Int)
    m = fill(val, size, size)
    for i in 1:size
        m[i, i] = 0.0
    end
    return m
end

begin
    # Random.seed!(1)
    
    ε = 1.0     # external stimulation weight
    λ = 0.10    # learning rate
    τ = 0.15    # softmax temperature
    wbounds = (0.25, 0.35)
    inhibition = 0.01
    T = 300     # number of simulation cycles

    nhidden = 10
    nfeatures = 10
    nlabels = nfeatures

    pools = (
        label = collect(1:nlabels),
        color = collect(1:nfeatures),
        shape = collect(1:nfeatures),
        orien = collect(1:nfeatures),
        hidden = collect(1:nhidden)
    )
    poolkeys = keys(pools)
    periphs = poolkeys[1:end-1]

    # Initialize vectors for network's values
    ext = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    net = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    pact = NamedTuple{poolkeys}(map(zeros ∘ length, pools))
    act = NamedTuple{poolkeys}(map(zeros ∘ length, pools))

    # Initialize weight matrices
    W = NamedTuple{periphs}(map(pool -> rand(Uniform(wbounds...), nhidden, length(pool)), pools[periphs]))
    M = NamedTuple{poolkeys}(map(pool -> inhmat(inhibition, length(pool)), pools))

    # Create data containers
    df = DataFrame(step=Int[], pool=String[], unit=Int[], net=Float64[], pact=Float64[], act=Float64[])
    wstore =  NamedTuple{poolkeys}([[] for k in poolkeys])

    # Define phases and events
    hebbian_phases = []#[20:30]#, 120:130]
    priming_phases = []#[20:30]#, 120:130, 220:T]
    binding_phases = []#[120:130]
    reset_steps = []#[31, 131]


    # Simulate n steps
    for step in 1:T
        # Clamp peripheral units familiarization
        if any([step in sp for sp in priming_phases])
            ext.label[1] = 1.0
        end

        # Clamp peripheral units for pairing
        if any([step in pp for pp in binding_phases])
            ext.label[1] = 1.0
            ext.color[1] = 1.0
            ext.shape[1] = 1.0
            ext.orien[1] = 1.0
        end

        # Activate hidden units
        net.hidden[:] = +([W[k] * act[k] for k in periphs]...) + ε .* ext.hidden
        # net.hidden[:] += rand(Normal(0, 0.001), nhidden)
        # pact.hidden[:] = softmax(net.hidden ./ τ)
        pact.hidden[:] = logistic.(net.hidden)
        act.hidden[:] = pact.hidden[:]
        # act.hidden[:] = oh(nhidden, sample(1:nhidden, Weights(pact.hidden)))
        
        # Activate peripheral units
        for k in periphs
            net[k][:] = W[k]' * act.hidden + ε .* ext[k]
            # net.hidden[:] += rand(Normal(0, 0.001), nfeatures)
            # pact[k][:] = softmax(net[k] ./ τ)
            pact[k][:] = logistic.(net[k])
            act[k][:] = pact[k][:]
            # act[k][:] = oh(nfeatures, sample(1:nfeatures, Weights(pact[k])))
        end

        # Hebbian learning
        if any([step in hp for hp in hebbian_phases])
            for k in poolkeys[1:end-1]
                W[k][:, :] += oja(W[k], pact.hidden, pact[k]; λ=λ)
            end
        end

        # Reset network
        if step in reset_steps
            for k in poolkeys
                ext[k][:] .= 0.0
                net[k][:] .= 0.0
                act[k][:] .= 0.0
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
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        ax2 = Axis(fig[i, 2], ylabel="pact($k)",
            limits=(
                (nothing, nothing),
                (-0.05, 1.05)
            )
        )
        ax3 = Axis(fig[i, 3], ylabel="weights ($k)",
            limits=(
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        ax4 = Axis(fig[i, 4], ylabel="active($k)",
            limits=(
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        ax5 = Axis(fig[i, 5], ylabel="Conflict($k)",
            limits=(
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        subdf = subset(df, :pool => ==(String(k)) |> ByRow)
        for gdf in groupby(subdf, :unit)
            lines!(ax1, 1:T, gdf.net, alpha=.4, linewidth=2)
            lines!(ax2, 1:T, gdf.pact, alpha=.4, linewidth=2)
        end
        if k != "hidden"
            for row in eachrow(hcat(wstore[k]...))
                lines!(ax3, 1:T, row, alpha=.7, linewidth=1)
            end
        end
        
        active = combine(groupby(subdf, [:step, :pool]), :act => argmax).act_argmax
        scatter!(ax4, 1:T, active, marker=:vline, color=:black)
        
        conflict = combine(groupby(subdf, [:step, :pool]), [:net, :pact] => ((net, prob) -> sum(net) * sum(prob .* log.(1 ./ prob))) => :conflict).conflict
        lines!(ax5, 1:T, conflict, color=:black, linewidth=2)
    end
    fig
end 

save("res2.png", fig)