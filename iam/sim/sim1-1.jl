# Inverted-U relationship between confidence and curiosity for general-knowledge questions.

using Pkg; Pkg.activate("IACTools")
using Revise

using AlgebraOfGraphics
using IAC
using IACTools
using Chain
using CSV
using DataFrames, DataFramesMeta
using DelimitedFiles
using Distributions
using GLMakie
using LogExpFunctions
using StatsBase


# Set up simulation params
simparams = (
    tol = 0.0001,
    T = 100,
    N = 20,
    Nq = 40,
)

# Construct test
testbase = CSV.read("iam/def/jets_sharks_table.csv", DataFrame)
testbank = []
for i in 1:nrow(testbase)
    for feature in names(testbase, Not(:id, :Name))
    # for feature in names(testbase, Not(:id, :firstName, :lastName))
        push!(testbank, 
            ([testbase[i, :Name]], feature, testbase[i, feature])
            # ([testbase[i, :firstName], testbase[i, :lastName]], feature, testbase[i, feature])
        )
    end
end
inds = sample(1:length(testbank), Weights(fill(1, length(testbank))), simparams.Nq)

# Define network hyperparameters
netparams = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    DECAY = 0.1, # Decay (must be between 0 and 1)
    EXT = 0.4, # External stimulation weight
    EXC = 0.1, # Excitation gain
    INH = 0.2, # Inhibition gain
    NOISE = .01, # Noise
    DMG = true,
    DMG_BREADTH = [b for b in LinRange(.10, .90, simparams.N)],
    DMG_LB = [0.3 for i in 1:simparams.N],
    DMG_UB = [1.0 for i in 1:simparams.N]
)

# Simulate (about 20 seconds with 20 participants and 40 questions)
df = DataFrame(subj=Int[], dmgOn=Bool[], dmgB=Float64[], dmgLB=Float64[], dmgUB=Float64[], qi=Int[], q=String[], answer=String[], correct=Bool[], confid=Float64[], unc=Float64[], fam=Float64[], cur=Float64[])
@time begin
for subj in 1:simparams.N
    # Instantiate network
    iacn = from_csv_table("iam/def/jets_sharks_table.csv", netparams; hidden_ids=:id)

    # Apply random damage to projections
    if netparams.DMG
        for proj in iacn.projections
            # find(iacn, "Name") in proj.dir && continue
            w, h = size(proj.mat)
            damage = rand(Bernoulli(netparams.DMG_BREADTH[subj]), w, h) .* rand(Uniform(netparams.DMG_LB[subj], netparams.DMG_UB[subj]), w, h)
            proj.mat[:, :] = max.(proj.mat .- damage, 0.0)
        end
    end

    # Administer test
    for ind in inds # item consists of query inputs (1) and queried feature (2)
        item = testbank[ind]
        inputs = item[1]
        feature = item[2]
        states = query(iacn, inputs, feature, simparams.T, simparams.tol)
        last_t = maximum(states.t)

        # Record confidence
        answer, confid = answer_confidence(states, feature)

        # Accuracy
        correct = answer == item[3]

        # Record conflict (curiosity)
        fam = @subset(states, :pool .== "hidden", :t .== last_t).act |> familiarity
        unc = @subset(states, :pool .== feature, :t .== last_t).act |> uncertainty
        cur = fam * unc

        push!(df, [subj, netparams.DMG, netparams.DMG_BREADTH[subj], netparams.DMG_LB[subj], netparams.DMG_UB[subj], ind, "$(feature)[$(inputs)]", answer, correct, confid, unc, fam, cur])
    end
end
end

# Relationship between raw variables
df = @chain df begin
    groupby(:subj)
    transform(
        :confid => (x -> (x .- mean(x)) ./ std(x)) => :confid_z,
        :cur => (x -> (x .- mean(x)) ./ std(x)) => :cur_z
    )
end
fig = Figure()
layer = data(df) * mapping(:confid, :cur; color=:dmgB) * (smooth() + visual(Scatter))
draw!(fig, layer)