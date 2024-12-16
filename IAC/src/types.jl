Name = Union{Symbol, S} where S <: AbstractString
NameVector = Union{Vector{S}, Vector{Symbol}} where S <: AbstractString

struct Pool{T <: AbstractFloat}
    name::Name
    size::Int
    unames::NameVector
    ext::Vector{T}
    net::Vector{T}
    act::Vector{T}
    con::Matrix{T}
    function Pool{T}(name::Name, size::Int; unames::Union{Missing, NameVector}=missing, con::Union{Missing, Matrix{T}}=missing) where {T <: Real}
        if ismissing(unames)
            unames = ["$(name)-u$(i)" for i in 1:size]
        end
        if size != length(unames)
            error("Problem building pool $name. Length of `unames` must be equal to pool size.")
        end
        if ismissing(con)
            con = zeros(T, size, size)
        end
        ext = zeros(T, size)
        net = zeros(T, size)
        act = zeros(T, size)
        return new(name, size, unames, ext, net, act, con)
    end
end

struct Proj{T}
    dir::Pair{Pool{Tₚ}, Pool{Tₚ}} where Tₚ <: AbstractFloat
    mat::Matrix{T}
end

mutable struct IACNetwork
    pools::Tuple{Vararg{Pool}}
    projections::Union{Proj, Tuple{Vararg{Proj}}}
    params::NamedTuple
    counter::Int
    function IACNetwork(pools::Tuple{Vararg{Pool}}, projections::Union{Proj, Tuple{Vararg{Proj}}}, params::NamedTuple)
        return new(pools, projections, params, 1)
    end
end