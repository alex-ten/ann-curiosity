# Effect of information-gap size on curiosity independent of confidence
using Revise
using AlgebraOfGraphics
using CSV
using CairoMakie
using DataFrames, DataFramesMeta
using Distributions
using StatsBase

using IAC
using IACTools

function mask_letters(word::S, indices::Vector{Int}) where S <: AbstractString
    return map(x -> x[1] in indices ? '_' : x[2], enumerate(word)) |> join
end

simparams = (
    tol = 0.0001,
    T = 50,
    N = 1,
    Nq = 100,
    Nw = 200,
    Lw = 7
)

# TODO Table construction code should go elsewhere!
df = @chain CSV.read("iam/def/unigram_freq.csv", DataFrame) begin
    @subset(length.(:word) .== simparams.Lw)
    @orderby(-:count)
    getindex(1:simparams.Nw, :)
end
new_df = DataFrame(
    :id => df.word,
    :word => df.word,
    [Symbol("l$i") => getindex.(rpad.(df.word, simparams.Lw), i) for i in 1:simparams.Lw]...
)
CSV.write("iam/def/words5_table.csv", new_df)
# TODO Table construction code should go elsewhere!

# Construct test
testbase = CSV.read("iam/def/words5_table.csv", DataFrame)
testbank = Dict(1 => [], 3 => [], 5 => [])
for m in [1, 3, 5]
    for i in 1:nrow(testbase)
        mask_inds = sample(1:simparams.Lw, m, replace=false)
        push!(testbank[m], [mask_letters(testbase[i, :word], mask_inds), testbase[i, :word]])
    end
end

# Simulate task
netparams = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    DECAY = 0.1, # Decay (must be between 0 and 1)
    EXT = 0.2, # External stimulation weight
    EXC = 0.1, # Excitation gain
    INH = 0.1, # Inhibition gain
    NOISE = 0.01, # Noise
)
iacn = from_csv_table("iam/def/words5_table.csv", netparams; hidden_ids=:id)

@time begin
    df = DataFrame(letters_missing=Int[], qi=Int[], q=String[], ans=String[], cor=Bool[], confid=Float64[], unc=Float64[], fam=Float64[], cur=Float64[])
    for m in [1, 3, 5]
        inds = sample(1:length(testbank[m]), Weights(fill(1, length(testbank[m]))), simparams.Nq)
        for ind in inds
            # Query
            problem, solution = testbank[m][ind]
            states = word_query(iacn, word_inp(problem), simparams.T, simparams.tol)
            last_t = maximum(states.t)

            # Guess
            ans, confid = answer_confidence(states, "word")

            # Accuracy
            cor = ans == solution

            # Conflict (curiosity)
            fam = @subset(states, :pool .== "hidden", :t .== last_t).act |> familiarity
            unc = @subset(states, :pool .== "word", :t .== last_t).act |> uncertainty
            cur = fam * unc
            push!(df, [m, ind, problem, ans, cor, confid, unc, fam, cur])
        end
    end
end


# Relationship between raw variables
fig = Figure()
layer = data(df) * mapping(:confid, :cur; color=:letters_missing => nonnumeric) * (smooth() + visual(Scatter))
draw(layer, legend=(position=:top, titleposition=:left, framevisible=true, padding=5))
fig