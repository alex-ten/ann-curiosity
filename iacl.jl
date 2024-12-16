using Distributions
using LinearAlgebra
using Random

Random.seed!(1)
rest = -0.1
amax = 1.0
amin = -0.2
decay = 0.1
ϵ = 0.4
α = 0.1
λ = 0.5

# Initialize IACL network
FUP = [zeros(6) for i in 1:12]
FRUP = fill(rest, 100)
W = [rand(Uniform(-0.05, 0.05), (100, 6)) for i in 1:12]

# Make 50 faces "known"
known = [rand(1:6, 12) for i in 1:50]

for i in 1:12
    for j in 1:50
        for k in known[j]
            W[i][j, :] .= -1.0
            W[i][j, k] = 1.0
        end
    end
end

i = 1
W[i] * known[i]