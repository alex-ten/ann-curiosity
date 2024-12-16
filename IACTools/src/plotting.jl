function i2c(idx::Int, width::Int; byrows::Bool=true)
    idx0 = idx - 1
    i = idx0 ÷ width
    j = idx0 % width
    !byrows && return j + 1, i + 1
    return i + 1, j + 1
end

function get_label(plot, index, position)
    x, y = position
    x = round(x) |> Int
    y = round(y; digits=4)
    
    label = "unit: $(plot.label[])\nact($x) = $y"
    return label
end

function mask_first(df::DataFrame, col::Symbol)
    dfcopy = copy(df)
    dfcopy[df.itert .== 1, col] .= NaN
    return dfcopy
end

function plot_pool_summary(df::DataFrame; select::Union{Missing, NameVector}=missing, figsize=missing)
    fig = Figure(size=ismissing(figsize) ? (400, 300) : figsize)
    plot_pool_summary!(fig[1, 1], df, spec; select=select)
    # DataInspector(fig)
    return fig
end

function plot_pool_summary!(grid_position::GridPosition, df::DataFrame, spec::Any; select::Union{Missing, NameVector}=missing)
    if !ismissing(select)
        df = subset(df, :pool => ByRow(in(select)))
    end
    function_name = (last ∘ last)(spec) # Symbol
    ax = Axis(grid_position,
            xlabel = "Step",
            ylabel = String(function_name),
            title = "Pool summary"
        )

    df1 = @chain subset(df, :pool => ByRow(in(pools))) begin
        groupby([:pool, :iter, :t])
        combine(spec)
    end
    for (i, gdf) in groupby(df1, :pool) |> enumerate
        lines!(ax, gdf.t, gdf[:, function_name], label=String(gdf.pool[1]), color=ColorSchemes.viridis.colors[i])
    end
    axislegend(ax, framevisible=false, position=:lt)
    return nothing
end

function plot_proj_state(proj::Proj)
    poola, poolb = proj.dir
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = String(poola.name),
        xticks = (1:poola.size, String.(poola.unames)), xticklabelrotation=π/2,
        ylabel = String(poolb.name),
        yticks = (1:poolb.size, String.(poolb.unames))
    )
    hm = heatmap!(ax, proj.mat, colormap=:binary, colorrange=(0.0, 1.0))
    Colorbar(fig[1, 2], hm)
    fig
end

function plot_proj_state(projs::Tuple{Vararg{Proj}}; ncols::Int=3, transp::Bool=false)
    fig = Figure(size=(1500, 900))
    for (idx, proj) in enumerate(projs)
        i, j = i2c(idx, ncols)
        poola, poolb = transp ? (proj.dir[2], proj.dir[1]) : proj.dir
        ax = Axis(fig[i, j],
            xlabel = String(poola.name),
            xticks = (1:poola.size, String.(poola.unames)), xticklabelrotation=π/2,
            ylabel = String(poolb.name),
            yticks = (1:poolb.size, String.(poolb.unames)),
            aspect = DataAspect()
        )
        m = transp ? tranpose(proj.mat) : proj.mat
        heatmap!(ax, m, colormap=:binary, colorrange=(0.0, 1.0))
    end
    Colorbar(fig[1:end, ncols+1], colorrange=(0.0, 1.0), colormap=:binary)
    DataInspector(fig)
    fig
end

function plot_proj_trace(history::Dict, pname::Name; ncols::Int=3, dir::Symbol=:in)
    fig = Figure()
    idx = 0
    for ((a, b), trace) in pairs(history)
        if b == pname
            idx += 1
            i, j = i2c(idx, ncols)
            ax = Axis(fig[i, j], title="$a")
            for receiver_idx in axes(trace, 2)
                for r in eachrow(trace[:, receiver_idx, :])
                    lines!(ax, r)
                end
            end
        end
    end
    return fig
end

function plot_iter_summary!(grid_position::GridPosition, df::DataFrame, spec::Any; limits=nothing, select::Union{Missing, NameVector}=missing, title=missing)
    grid = GridLayout(grid_position)
    if !ismissing(select)
        df = subset(df, :pool => ByRow(in(select)))
    end
    function_label = (last ∘ last)(spec) # Symbol
    df1 = combine(groupby(df, [:iter, :pool, :unit]), spec)
    for (idx, gdf) in groupby(df1, :pool) |> enumerate
        ax = Axis(grid[idx, 1],
            # limits = limits,
            title = ismissing(title) ? String(gdf.pool[1]) : title,
            ylabel = function_label,
            xlabel = "Episode"
        )
        for udf in groupby(gdf, :unit)
            scatterlines!(ax, udf.iter, udf[:, function_label], label=String(udf.unit[1]))
        end
    end
    return nothing
end

function acts_plot_layers(df::DataFrame)
    layer1 = data(df) *
    mapping(:t, :act; color=:unit, layout=:pool) *
    visual(Lines)

    last_vals = @chain groupby(df, [:pool]) begin
        @subset(:t .== maximum(:t))
    end
    layer2 = data(last_vals) * 
    mapping(:t, :act, text=:unit=>verbatim, color=:unit, layout=:pool) *
    visual(Makie.Text, fontsize=12, align=(:right, :bottom))

    return layer1 + layer2
end

function plot_acts(df::DataFrame)
    fig = Figure()
    layers = acts_plot_layers(df)
    # lims = (nothing, (iacn.params.MIN, iacn.params.MAX))
    lims = (nothing, nothing)
    draw!(fig, layers, axis=(; limits=lims))
    return fig
end