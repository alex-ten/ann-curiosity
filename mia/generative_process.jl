using Pkg; Pkg.activate("IACTools")
using Distributions
using InvertedIndices
using Revise

includet("font.jl")
includet("utils.jl")

WORDS = ["ON", "OH"]
LETTERS = ['O', 'N', 'H']
FEATURES = collect(1:14)

nwords = length(WORDS)
p = fill(1 / nwords, nwords)
dwords = Categorical(p)

W = WORDS[rand(dwords)] |> Word
L = [ifelse(rand() < 0.9, l, rand(LETTERS[findfirst(==(l), LETTERS) |> Not])) for l in W.letters]
for l in W.letters
    if rand() < 0.9
        println(findfirst(==(l), LETTERS))
        ii = findfirst(==(l), LETTERS) |> Not
        println(ii)
        println(LETTERS[ii] |> rand)
    else
        println(l)
    end
end

