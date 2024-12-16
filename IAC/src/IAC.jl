module IAC

using DataFrames
using DelimitedFiles
using Distributions

# Types
include("types.jl")
export Name, NameVector
export IACNetwork
export Pool
export Proj

# IAC processing routines
include("routines.jl")
export clamp_pool!, unclamp_pool!
export clamp_unit!, unclamp_unit!
export unclamp_all!
export log_state
export reset!
export step!
export nsteps!

# Helper functions
include("helpers.jl")
export fill_hollow
export find
export indexinpool

# Base methods
include("base.jl")
export show

end