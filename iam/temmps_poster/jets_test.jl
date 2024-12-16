using Pkg; Pkg.activate("IACTools")
using Revise

using IAC
using IACTools
using Chain
using CSV
using DataFrames

import Random: shuffle!
import StatsBase: sample


params = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    γ = 0.1,     # decay (must be between 0 and 1)
    ε = 0.4,     # external stimulation weight
    α = 0.1,     # Excitation gain
    β = 0.1,     # Inhibition gain
    σ = 1.0    # noise
)
iacn = from_def("jets/jets_pools_def", "jets/jets_proj_def", params)

# Test
begin
    # Construct test
    n, m = 5, 2
    material = CSV.read("jets/jets_sharks_table.csv", DataFrame)
    ids = sample(1:nrow(material), n; replace=false)
    features = names(material)
    items = []
    for id in ids
        for feature in sample(features[2:end], m; replace=false)
            push!(items, (Symbol(material[id, :Name]), Symbol(feature), material[id, feature]))
        end
    end
    shuffle!(items)

    # Administer test
    score = 0
    for (i, item) in enumerate(items)
        person, feature, ans = item
        println("$i) What is $(person)'s $(feature) [$ans]?")
        resp, dur = query(iacn, person, feature ; nmax=150, abstol=0.01)
        reset!(iacn)
        correct = String(ans) == String(resp)
        score += correct
        mark = correct ? '✅' : '❌'
        println("... $dur steps later, response is \"$resp\" $mark")
    end
    println("\nFinal score is $score/$(n * m)")
end