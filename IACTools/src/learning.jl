function scale_proj!(network::IACNetwork, pup1::Pair{N, N}, pup2::Pair{N, N}, s::R) where {R <: Real, N <: Name}
    wmax = 1.0
    pnames = map(first, [pup1, pup2])
    unames = map(last, [pup1, pup2])
    for proj in network.projections
        if map((in ∘ names)(proj), pnames) |> all
            a, b = proj.dir
            i = filter(!isnothing, indexin(unames, a.unames))[1]
            j = filter(!isnothing, indexin(unames, b.unames))[1]
            proj.mat[i, j] += (wmax - proj.mat[i, j])*s
            return nothing
        end
    end
    throw("Couldn't find projection between $pup1 and $pup2. Nothing changed.")
end

function explc(t::R) where R <: Real
    return 1 - exp(l*t)
end

function increment_proj!(network::IACNetwork, pup1::Pair{N, N}, pup2::Pair{N, N}, inc::R) where {R <: Real, N <: Name}
    wmax = 1.0
    pnames = map(first, [pup1, pup2])
    unames = map(last, [pup1, pup2])
    for proj in network.projections
        if map((in ∘ names)(proj), pnames) |> all
            a, b = proj.dir
            i = filter(!isnothing, indexin(unames, a.unames))[1]
            j = filter(!isnothing, indexin(unames, b.unames))[1]
            proj.mat[i, j] += (wmax - proj.mat[i, j])*inc
            return nothing
        end
    end
    throw("Couldn't find projection between $pup1 and $pup2. Nothing changed.")
end