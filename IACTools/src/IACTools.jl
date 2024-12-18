module IACTools

using AlgebraOfGraphics
using IAC
using Chain
using ColorSchemes
using CSV
using DataFrames, DataFramesMeta
using CairoMakie
using LogExpFunctions
using StatsBase

# Building from files
include("building.jl")
export from_def
export from_csv_table
export from_df

# Plotting
include("plotting.jl")
export align_records
export mask_first
export plot_acts
export plot_pool_summary!
export plot_iter_summary!
export plot_proj_state
export plot_proj_trace
export get_label
export scale
export acts_plot_layers

# Theory 1: processing
include("monitoring.jl")
export conflict
export familiarity
export uncertainty
export confidence

# Theory 2: learning
include("learning.jl")
export increment_proj! 

# Testing
include("testing.jl")
export answer_confidence
export query, word_query
export word_inp

# Other
export mask_letters

function mask_letters(word::S, indices::Vector{Int}) where S <: AbstractString
    return map(x -> x[1] in indices ? '_' : x[2], enumerate(word)) |> join
end

# function forget(hidden_id::Name)


end
