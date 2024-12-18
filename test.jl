# Load local modules
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using IAC




# Create a simple test
# Let's create a Pool (assuming it takes some parameters)
pool = Pool(10)  # You might need different parameters based on your actual implementation

# Try one of the exported functions
clamp_pool!

# Print some information
println("Pool state after clamping: ", pool)