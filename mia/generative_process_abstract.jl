using Pkg; Pkg.activate("MIA")
using Distributions
using InvertedIndices
using Revise


includet("font.jl")
includet("utils.jl")

vocab = Vocab(["FOX", "CAT", "RAT", "BAT", "DOG", "BUG"])

word = rand(vocab)
letters = rand(font, word)