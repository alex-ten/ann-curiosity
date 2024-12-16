function fill_hollow(val::T, dim::Int) where T <: Real
    m = fill(val, dim, dim)
    for i in 1:dim
        m[i, i] -= val
    end
    return m
end

find(network::IACNetwork, pool_name::Name) = filter(pool -> pool.name==pool_name, network.pools)[1]

# function scale_proj!(network::IACNetwork, pname1::Name, pname2::Name, s::T) where T <: Real
#     pool1 = find(network, pname1)
#     pool2 = find(network, pname2)
#     for proj in network.projections
#         if (pool1 in proj.dir) & (pool2 in proj.dir)
#             proj.mat[:, :] = proj.mat .* s
#             break
#         end
#     end
#     return nothing
# end

indexinpool(pool::Pool, uname::Name) = findfirst(==(uname), pool.unames)
