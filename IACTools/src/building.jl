function make_unique(vec::Vector{S}) where S <: AbstractString
    count_dict = Dict{String, Int}()
    
    return map(vec) do item
        if haskey(count_dict, item)
            count_dict[item] += 1
            return item * string(count_dict[item])
        else
            count_dict[item] = 0
            return item
        end
    end
end


function from_def(pools_def::String, proj_def::String, params::NamedTuple)
    # Construct pools
    iac_pool_params = readdlm(pools_def, ',', String, '\n')[:, 2:end]
    pool_unames = iac_pool_params[:, 1]
    pool_names = iac_pool_params[:, 2]

    pools = []
    for name in unique(iac_pool_params[:, 2])
        a = indexin([name], pool_names)[1]
        s = sum(name .== pool_names)
        b = a + s - 1
        pool = Pool{Float32}(name, s; unames=pool_unames[a:b], con=fill_hollow(Float32(-1.0), s))
        push!(pools, pool)
    end

    # Construct projections
    constr_hid_vis = readdlm(proj_def, ' ', Float64, '\n')
    projections = []
    for pool in pools[1:end-1]
        a = indexin([pool.name], pool_names)[1]
        s = sum(pool.name .== pool_names)
        b = a + s - 1
        push!(projections, Proj(pool => pools[end], constr_hid_vis[a:b, :]))
    end

    # Combine pools and weights into a network
    iacn = IACNetwork(Tuple(pools), Tuple(projections), params)
    reset!(iacn; reset_counter=true)

    return iacn
end

function from_csv_table(path::String, params::NamedTuple; hidden_ids::Union{Name, Missing}=missing)
    df = CSV.read(path, DataFrame)
    return from_df(df, params; hidden_ids=hidden_ids)
end

function from_df(df::DataFrame, params::NamedTuple; hidden_ids::Union{Name, Missing}=missing)
    # Construct input pools
    pools = Pool[]
    for name in names(df)
        name == String(hidden_ids) && continue
        u = unique(df[:, name])
        u = u[.!ismissing.(u)]
        s = length(u)
        pool = Pool{Float32}(name, s; unames=String.(u), con=fill_hollow(Float32(-1.0), s))
        push!(pools, pool)
    end

    # Add hidden pool
    let
        unames = df[:, hidden_ids] |> make_unique
        unames = unames[.!ismissing.(unames)]
        unames = ismissing(hidden_ids) ? missing : "_" .* String.(unames) .* "_"
        s = nrow(df)
        pool = Pool{Float32}("hidden", s; unames=unames, con=fill_hollow(Float32(-1.0), s))
        push!(pools, pool)
    end

    # Construct projections
    projections = []
    for pool in pools[1:end-1]
        m = zeros(Float32, pool.size, last(pools).size)
        for (i, uname) in names(pool) |> enumerate
           m[i, :] .= Float32.(map(x -> ismissing(x) ? false : x .== String(uname), df[:, pool.name]))
        end
        push!(projections, Proj(pool => pools[end], m))
    end

    # Combine pools and weights into a network
    iacn = IACNetwork(Tuple(pools), Tuple(projections), params)
    reset!(iacn; reset_counter=true)

    return iacn
end