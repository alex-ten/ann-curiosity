"""A convencience function to pose a specific query targeting a feature value of a concept activated by inputs."""
function query(network::IACNetwork, inputs::NameVector, nmax::Int, abstol::T) where T <: Real
    reset!(network; reset_counter=true)
    for uname in inputs
        clamp_unit!(network, uname)
    end
    # clamp_pool!(network, feature; distribute=true)
    # nsteps!(network, 100, abstol; logstates=true)
    # unclamp_all!(network)
    states = nsteps!(network, nmax, abstol; logstates=true)
    return states
end

function answer_confidence(states::DataFrame, feature::Name)
    conf_df = confidence(states, feature)
    ismissing(conf_df) && return "", 0.0
    max_row = conf_df[argmax(conf_df.confidence), :]
    ans = max_row.unit
    conf = max_row.confidence
    return ans, conf
end

function word_inp(s::String)
    return [string(c) for c in s]
end

function word_query(network::IACNetwork, inps::NameVector, nmax::Int, abstol::T) where T <: Real
    reset!(network; reset_counter=true)
    for (ind, letter) in enumerate(inps)
        letter == "_" && continue
        clamp_unit!(find(network, "l$ind"), letter, network.params.MAX)
    end
    states = nsteps!(network, nmax, abstol; logstates=true)
    return states
end