"""Computes activation update (from McClelland's PDP handbook)"""
function Δa(pool::Pool; rest::T=-0.1, maxact::T=1.0, minact::T=-0.2, γ::T=0.1) where T <: Real
    Δ = Vector{Float64}(undef, pool.size)
    for i in 1:length(Δ)
        if pool.net[i] > 0
            Δ[i] = pool.net[i] * (maxact - pool.act[i]) - γ * (pool.act[i] .- rest)
        else
            Δ[i] = pool.net[i] * (pool.act[i] - minact) - γ * (pool.act[i] .- rest)
        end
    end
    return Δ
end

"""Bidirectionally propagates activation values through connection weights between units of two pools"""
function propagate!(proj::Proj, α::T) where T <: Real
    # Forward feed (from A to B, given A => B)
    first(proj.dir).net[:] += (α .* proj.mat) * max.(0, last(proj.dir).act)

    # Backward feed (from B to A, given A => B)
    last(proj.dir).net[:] += (α .* transpose(proj.mat)) * max.(0, first(proj.dir).act)
    return nothing
end

"""Updates activation value of each unit of each pool in the network."""
function update!(network::IACNetwork)
    for pool in network.pools
        pool.act[:] += Δa(pool;
            rest = network.params.REST,
            maxact = network.params.MAX,
            minact = network.params.MIN,
            γ = network.params.DECAY
        )
    end
    return nothing
end

"""Executes one cycle of IAC processing, which entails the following steps:
1. Net inputs are reset.
2. Nonnegative activations between all connected pools are propagated (between-pool excitation).
3. Nonnegative activations within pools are propagated (within-pool inhibition).
4. External excitation is added to net inputs.
5. Random noise is added to net inputs.
6. Finally, `update!` is called on the network and the network counter is incremented.
"""
function step!(network::IACNetwork)
    for pool in network.pools
        pool.net[:] .= 0.0
    end

    # Accumulate projected inputs (updates net input of senders and receivers of each proj)
    for proj in network.projections
        propagate!(proj, network.params.EXC)
    end

    # Accumulate inhibitory input and external input
    for pool in network.pools
        pool.net[:] += (network.params.INH .* pool.con) * max.(0, pool.act)
        pool.net[:] += pool.ext .* network.params.EXT
        if network.params.NOISE > 0
            pool.net[:] += rand(Normal(0, network.params.NOISE), pool.size)
        end
    end

    # Compute activation of each unit in each pool
    update!(network)
    network.counter += 1
end

"""Executes `step!` `T` times. If `logstates==true`, the function will return a DataFrame containing a record of the network state at each time step."""
function nsteps!(network::IACNetwork, nmax::Int; logstates::Bool=false)
    logs = DataFrame[]
    for t in 1:nmax
        step!(network)
        logstates && push!(logs, log_state(network, t))
    end
    return logstates ? vcat(logs...) : nothing
end

"""Executes `step!` until convergence or until `nmax` is reached."""
function nsteps!(network::IACNetwork, nmax::Int, abstol::T; logstates::Bool=false) where T <: Real
    logs = DataFrame[log_state(network, 1)]
    for t in 2:nmax+1
        step!(network)
        logstates && push!(logs, log_state(network, t))
        converged(logs[t-1], logs[t]; abstol) && break
    end
    return logstates ? vcat(logs...) : nothing
end

""""""
function converged(slog0::DataFrame, slog1::DataFrame; abstol::T) where T <: Real
    df = slog1[:, :]
    df[!, :act_prev] = slog0.act
    df[!, :abs_conv] = abs.(df.act_prev - df.act) .< abstol
    gdf = groupby(df, :pool)
    abs_test = all(combine(gdf, :abs_conv => all; renamecols=false).abs_conv)
    return abs_test
end

abstol(a, b, tol) = abs.(a, b) .< tol
reltol(a, b, tol) = abs.(a - b) .< tol .* max.(abs.(a), abs.(b))

"""Sets the external activation of a unit (identified by `uname`) inside a `pool`` to `val`"""
function clamp_unit!(pool::Pool, uname::Name, val::T) where T <: Real
    pool.ext[findfirst(==(uname), pool.unames)] = val
    return nothing
end

"""Sets the external activation of a unit (identified by `uname`) inside a the network to network's maximum activation value (`params.MAX`) or the value provided"""
function clamp_unit!(network::IACNetwork, uname::Name, val::Union{Missing, T}=missing) where T <: Real
    for pool in network.pools
        if uname in pool.unames
            pool.ext[findfirst(==(uname), pool.unames)] = ismissing(val) ? network.params.MAX : val
        end
    end
    return nothing
end
unclamp_unit!(network::IACNetwork, uname::Name) = clamp_unit!(network, uname, 0.0)

"""Sets the external activation in the entire pool (identified by `pname` to `val` or the maximum activation value of the network (`params.MAX`)"""
function clamp_pool!(network::IACNetwork, pname::Name, val::Union{Missing, T}=missing; distribute::Bool=false) where T <: Real
    val_ = ismissing(val) ? network.params.MAX : val
    for pool in network.pools
        if pool.name == pname
            pool.ext[:] .= distribute ? val_ / pool.size : val_
        end
    end
    return nothing
end
unclamp_pool!(network::IACNetwork, pname::Name) = clamp_pool!(network, pname, 0.0)

"""Unclamp all inputs"""
function unclamp_all!(network::IACNetwork)
    for pool in network.pools
        pool.ext[:] .= 0.0
    end
end

"""Resets pools in the network, setting external activation and net input to 0 and activation to `REST`"""
function reset!(network::IACNetwork; reset_counter::Bool=false)
    network.counter = reset_counter ? 1 : network.counter
    for pool in network.pools
        pool.ext[:] .= 0.0
        pool.net[:] .= 0.0
        pool.act[:] .= network.params.REST
    end
end

"""Reads the current state of network's units and stores them inside a DataFrame."""
function log_state(network::IACNetwork, iter::Int)
    dfs = DataFrame[]
    (t=Int[], pool=String[], unit=String[], ext=Float16, net=Float16[], act=Float16[])

    for pool in network.pools
        push!(dfs, DataFrame(
            t = fill(network.counter, pool.size),
            itert = fill(iter, pool.size),
            pool = fill(pool.name, pool.size),
            unit = pool.unames,
            ext = pool.ext,
            net = pool.net,
            act = pool.act
        ))
    end

    return vcat(dfs...)
end