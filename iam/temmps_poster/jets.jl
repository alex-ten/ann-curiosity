import Pkg; Pkg.activate("IAC")
using Revise
using IAC
using DelimitedFiles

# Construct pools
vis_names = readdlm("pdptool/iac/jetsvisunames", '-', String, '\n')[:, 2] .|> Symbol
vis = Pool{Float64}("visible", length(vis_names); unames=vis_names)
vis.con[:, :] = readdlm("pdptool/iac/constrvis", ' ', Float64, '\n')

hid_names = readdlm("pdptool/iac/jetshidunames", '_', String, '\n')[:, 2] .|> Symbol
hid = Pool{Float64}("hidden", length(hid_names); unames=hid_names)
hid.con[:, :] = fill_hollow(-1.0, hid.size)

# Construct (read) weight matrices
W = Proj(
    vis => hid,
    readdlm("pdptool/iac/constr_hid_vis", ' ', Float64, '\n')
)

# (Currupt weights)
W.mat[:] = W.mat .* .3

# Combine pools and weights into a network
params = (
    MAX = 1.0,
    MIN = -0.2,
    REST = -0.1,
    γ = 0.1,     # decay (must be between 0 and 1)
    ε = 0.4,     # external stimulation weight
    α = 0.1,     # Excitation gain
    β = 0.1,     # Inhibition gain
    σ = 0.0001   # noise
)
iacn = IACNetwork((vis, hid), (W,), params)

# Simulate
reset!(iacn)
clamp_unit!(vis, :Nick, iacn.params.MAX)
states = step!(iacn, 50; logstates=true)

# Visualize activations
plotacts(states; limits=((nothing, nothing), (params.MIN, params.MAX)))