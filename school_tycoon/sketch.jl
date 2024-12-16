using Pkg; Pkg.activate("IACTools")
using DataFrames
using Distributions
using StatsBase


mutable struct Agent
    name::String
    network::Vector{Agent}
    ties::Vector{R} where R <: Real
    target::Union{Agent, Nothing}
    intraversion::R where R <: Real
    function Agent(name::String, intraversion::Float64)
        return new(name, Agent[], Float64[], nothing, intraversion)
    end
end

function Base.show(io::IO, agent::Agent)
    print(io, "å $(agent.name)")
end

function intermingle!(agents::Vector{Agent})
    l = length(agents) - 1
    for (i, a) in enumerate(agents)
        a.network = agents[Not(i)]
        a.ties = ones(l)
    end
end

function set_target!(a)
    ix = sample(1:length(a.network)+1, vcat(a.ties, [a.intraversion]) |> Weights)
    a.target = ix <= length(a.network) ? a.network[ix] : nothing
    return a.target
end

function chat!(a1::T, a2::T) where T <: Agent
    println("$a1 and $a2 are chatting!")
end

agents = [Agent(name, 100.0) for name in ("Alice", "Bob", "Charlie", "Dan", "Eddie")]

intermingle!(agents)

for t in 1:10
    for a in agents
        set_target!(a)
    end

    for a in agents
        a.target |> isnothing && continue
        a.target.target == a && chat!(a, a.target)
    end
end