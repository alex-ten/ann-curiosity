function Base.show(io::IO, pool::Pool)
    print(io, "Pool($(pool.name), $(pool.size))")
end

function Base.show(io::IO, network::IACNetwork)
    pools = network.pools
    npools = length(pools)
    poolnames = sort([pool.name for pool in pools])
    print(io, "IACNetwork($poolnames, $npools)")
end

function Base.show(io::IO, proj::Proj)
    println(io, "$(first(proj.dir)) # $(last(proj.dir))")
end

function Base.names(network::IACNetwork)
    return [pool.name for pool in network.pools]
end

function Base.names(pool::Pool)
    return pool.unames
end

function Base.names(proj::Proj)
    return [pool.name for pool in proj.dir]
end