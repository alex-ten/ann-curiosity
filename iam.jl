using Pkg; Pkg.activate("Familiarity")
using CairoMakie
using DataFrames
using Distributions
using InvertedIndices: Not
using LinearAlgebra
using LogExpFunctions: softmax, logistic
using Random
using StatsBase: sample, Weights

function Δa(act::Vector{T}, net::Vector{T}; rest::T=-0.1, mx::T=1.0, mn::T=-0.2, γ::T=0.1) where T <: Real
    Δ = Vector{Float64}(undef, length(net))
    for i in 1:length(Δ)
        if net[i] > 0
            Δ[i] = net[i] * (mx - act[i]) - γ .* (act[i] .- rest)
        else
            Δ[i] = net[i] * (act[i] - mn) - γ .* (act[i] .- rest)
        end
    end
    return Δ
end

function Δw(W::Matrix{T}, enact::Vector{T}, exact::Vector{T}; λ::T) where T <: AbstractFloat
    aᵢaⱼ = enact * exact'
    adj = zero(aᵢaⱼ) .+ 1.0
    adj[aᵢaⱼ .> 0] .= -1.0
    Δ = λ .* aᵢaⱼ .* (1 .+ W .* adj)
    return Δ
end

function oh(size::Int, i::Int)
    v = zeros(Int, size)
    v[i] = 1
    return v
end

function oja(W::Matrix{T}, enact::Vector{T}, exact::Vector{T}; λ::T) where T <: AbstractFloat
    return λ .* enact * (exact - W' * enact)'
end

function inhmat(val::Float64, size::Int)
    m = fill(-val, size, size)
    for i in 1:size
        m[i, i] = 0.0
    end
    return m
end

begin
    Random.seed!(1)
    MAX = 1.0
    MIN = 0.0
    REST = 0.0
    γ = 0.9     # decay (must be between 0 and 1)
    ε = 1.0     # external stimulation weight
    λ = 0.10    # learning rate
    σ = 0.0001  # noise to net input

    wbounds = (0.25, 0.35)
    inhibition = 0.01
    T = 5     # number of simulation cycles

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
    act = NamedTuple{poolkeys}(map(zeros ∘ length, pools))

    # Initialize weight matrices
    W = NamedTuple{periphs}(map(pool -> rand(Uniform(wbounds...), nhidden, length(pool)), pools[periphs]))
    M = NamedTuple{poolkeys}(map(pool -> inhmat(inhibition, length(pool)), pools))

    # Create data containers
    df = DataFrame(step=Int[], pool=String[], unit=Int[], net=Float64[], act=Float64[])
    wstore =  NamedTuple{poolkeys}([[] for k in poolkeys])

    # Define phases and events
    hebbian_phases = []#, 120:130]
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
            ext.label[1] = MAX
            ext.color[1] = MAX
            ext.shape[1] = MAX
            ext.orien[1] = MAX
        end

        # Activate hidden units
        net.hidden[:] = +([W[k] * act[k] for k in periphs]...) + ε .* ext.hidden + M.hidden * act.hidden
        net.hidden[:] += rand(Normal(0, σ), nhidden)
        act.hidden[:] = Δa(act.hidden, net.hidden; rest=REST, mx=MAX, mn=MIN, γ=γ)
        
        # Activate peripheral units
        for k in periphs
            net[k][:] = W[k]' * act.hidden + ε .* ext[k] + M[k] * act[k]
            net.hidden[:] += rand(Normal(0, σ), nfeatures)
            act[k][:]+= Δa(act[k], net[k]; rest=REST, mx=MAX, mn=MIN, γ=γ)
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
            for (i, (n, a)) in enumerate(zip(net[k], act[k]))
                push!(df, [step, String(k), i, n, a])
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
        ax2 = Axis(fig[i, 2], ylabel="act($k)",
            limits=(
                (nothing, nothing),
                (MIN-0.05, MAX+0.05)
            )
        )
        ax3 = Axis(fig[i, 3], ylabel="weights ($k)",
            limits=(
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        ax5 = Axis(fig[i, 4], ylabel="Conflict($k)",
            limits=(
                (nothing, nothing),
                (nothing, nothing)
            )
        )
        subdf = subset(df, :pool => ==(String(k)) |> ByRow)
        for gdf in groupby(subdf, :unit)
            lines!(ax1, 1:T, gdf.net, alpha=.4, linewidth=2)
            lines!(ax2, 1:T, gdf.act, alpha=.4, linewidth=2)
        end
        if k != "hidden"
            for row in eachrow(hcat(wstore[k]...))
                lines!(ax3, 1:T, row, alpha=.7, linewidth=1)
            end
        end
        
        # conflict = combine(groupby(subdf, [:step, :pool]), [:net, :act] => ((net, prob) -> sum(net) * sum(prob .* log.(1 ./ prob))) => :conflict).conflict
        # lines!(ax5, 1:T, conflict, color=:black, linewidth=2)
    end
    fig
end 

# save("res2.png", fig)


begin
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="net input", ylabel="Δa")
    x = LinRange(-1000, 1000, 200) |> collect
    for a in [REST, .2 * MAX, .5 * MAX, .7 * MAX, MAX]
        act = zero(x) .+ a
        y = Δa(act, x; rest=REST, mx=MAX, mn=MIN, γ=γ)
        lines!(ax, x, y)
    end
    fig2
end