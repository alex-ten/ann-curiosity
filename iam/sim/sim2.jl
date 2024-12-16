# Effect of cue familiarity on search duration

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
using Random
using StatsBase

# Prepare mushrooms dataset (jets and sharks is too small and features are too familiar)

# Simulation parameters
simparams = (
    tol = 0.01,
    T = 500,
    N = 1,
    Nq = 100,
    Nw = 200,
    Lw = 7
)

# Construct test
testbase = CSV.read("iam/def/jets_sharks_table.csv", DataFrame)
testbank = []
for (i, r) in enumerate(eachrow(shuffle(testbase)))
    inputs = r[2:end] |> collect .|> String
    item = (inp=inputs, feature="Name", answer=String(r.Name), fam=i <= 14 ? 1 : 0, hidden_id=String(r.id))
    push!(testbank, item)
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
    NOISE = 0.0001, # Noise
    PROJ_NOISE = Normal(0, 0.005)
)

# Create network
iacn = from_csv_table("iam/def/jets_sharks_table.csv", netparams; hidden_ids=:id)

# Reset out all projections
for proj in iacn.projections
    proj.mat[:, :] .= 0.0
end

# Implement familiarity manipulation (SIMPLIFY!)
h = find(iacn, "hidden")
for item in testbank
    if item.fam == 1
        hidden_ind = indexinpool(h, "_" * item.hidden_id * "_")
        for proj in iacn.projections
            for feature in item.inp
                feature_pool = first(proj.dir)
                if feature in feature_pool.unames
                    feature_ind = indexinpool(feature_pool, feature)
                    proj.mat[feature_ind, hidden_ind] = .3
                end
            end
        end
    end
end

# Add random noise to connections
for proj in iacn.projections
    noisy = proj.mat + rand(netparams.PROJ_NOISE, size(proj.mat)...)
    proj.mat[:, :] = max.(min.(noisy, 1.0), 0.0)
end

plot_proj_state(iacn.projections)

# Administer test
@time begin
    df = DataFrame(qi=Int[], q=String[], seen=Bool[], ans=String[], cor=Bool[], confid=Float64[], unc=Float64[], fam=Float64[], cur=Float64[], nsteps=Int[])
    for (i, item) in enumerate(testbank)
        states = query(iacn, item.inp, simparams.T, simparams.tol)
        last_t = maximum(states.t)

        # Guess
        ans, confid = answer_confidence(states, item.feature)

        # Accuracy
        cor = ans == item.answer

        # Conflict (curiosity)
        fam = @subset(states, :pool .== "hidden", :t .== last_t).act |> familiarity
        unc = @subset(states, :pool .== item.feature, :t .== last_t).act |> uncertainty
        cur = fam * unc
        push!(df, [i, join(item.inp, '-'), item.fam, ans, cor, confid, unc, fam, cur, last_t])
    end
end

scatter(df.fam, df.nsteps)
scatter(df.fam, df.cur)