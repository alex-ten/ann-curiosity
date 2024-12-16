function scale(x::Any, maxim::R=1.0) where R <: Real
    minim = minimum(x)
    return (x .- minim) ./ (maxim - minim)
end

function familiarity(x)
    return maximum(scale(x))
end

function uncertainty(x)
    p = softmax(x ./ .12)
    return -sum(p .* log.(p))
end

# conflict(familiarity, uncertainty) = familiarity * uncertainty

function conflict(x; component::Symbol=:both)
    nonnegx = max.(0.0, x)
    s = sum(nonnegx)
    component == :sum && return s
    p = softmax(x ./ 0.01)
    H = -sum(p .* log.(p))
    component == :uncertainty && return H
    return  s * H
end

"""Read confidence about a response"""
function confidence(states::DataFrame, feature::Name)
    T = maximum(states.t)
    active = @subset(states, :pool .== feature, :act .> 0.0)
    isempty(active) && return missing
    return @chain active begin
        groupby(:t)
        transform(:act => softmax)
        groupby(:unit)
        # @aside println(_)
        combine(:act_softmax => (x -> sum(x) / T) => "confidence")
    end
end


function read_fam()
end

