import Pkg; Pkg.activate("IACTools")
using Revise
using IAC
using DelimitedFiles
using GLMakie

iac_pool_params = readdlm("/Users/alexten/Projects/mia/jets/jets_pools_def", ',', String, '\n')[:, 2:end] .|> Symbol
pool_unames = iac_pool_params[:, 1]
vis_names = String.(pool_unames[1:end-27])
hid_names = String.(pool_unames[42:end])
m = readdlm("/Users/alexten/Projects/mia/jets/jets_proj_def_corrupted", ' ', Float64, '\n')

begin
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Visible units",
        xticks = (1:length(vis_names), vis_names), xticklabelrotation=π/2,
        ylabel = "Hidden units",
        yticks = (1:length(hid_names), hid_names)
    )
    heatmap!(ax, m, colormap=:binary)
    fig
end