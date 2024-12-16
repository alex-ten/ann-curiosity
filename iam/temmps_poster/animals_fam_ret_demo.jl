using Pkg; Pkg.activate("IACTools")
using Revise

using IAC
using IACTools
using Chain
using DataFrames
using DelimitedFiles
# using GLMakie
using CairoMakie

function runsim(network::IACNetwork, T::Int, abstol::Float64)
    reset!(network, reset_counter=true)
    clamp_unit!(network, :Civet)
    clamp_pool!(network, :Diet)
    return nsteps!(network, T, abstol; logstates=true)
end

begin
    # Define network hyperparameters
    params = (
        MAX = 1.0,
        MIN = -0.2,
        REST = -0.1,
        DECAY = 0.1, # decay (must be between 0 and 1)
        EXT = 0.1*.2, # external stimulation weight
        EXC = 0.1*3, # Excitation gain
        INH = 0.1, # Inhibition gain
        NOISE = 0.0001*.5 # noise
    )
    # Factorial design
    factors = (
        repetitions = (
            few = 2,
            fewer = 4,
            many = 8
        ),
        facts = (
            fewer = (:Activity => :Nocturnal,),
            many = (:Class=>:Mammal, :Habitat=>:Terrestrial, :Activity=>:Nocturnal, :Continent=>:Americas)
        )
    )
        
    T = 1000
    tol = 0.0001
    data = []
    for nreps in factors.repetitions
        for facts in factors.facts
            iacn = from_def("jets/animals_pools_def", "jets/jets_proj_def", params)
            for proj in iacn.projections
                proj.mat[:, :] .= 10e-5
            end
            for rep in 1:nreps
                increment_proj!(iacn, :Species => :Civet, :Hidden => :_Civet_, 0.12)
            end
            for fact in facts
                increment_proj!(iacn, fact, :Hidden => :_Civet_, 0.5)
            end
            increment_proj!(iacn, :Diet => :Carnivore, :Hidden => :_Civet_, 0.1)
            states = runsim(iacn, T, tol)
            states[!, :nreps] .= nreps
            states[!, :nfacts] .= length(facts)
            push!(data, states)
        end
    end
    df = vcat(data...)
end

theme = Theme(
    fontsize=50,
    Axis = (
        spinewidth=3,
        xgridwidth=3,
        ygridwidth=3
    ),
    Lines = (
        linewidth=10,
    ),
    Legend = (
        framevisible=false,
        labelsize=40,
        linewidth=8,
        patchlabelgap=20,
        patchsize=(70, 20)
    )
)

begin
    z = .75
    fig = with_theme(theme) do
        fig = Figure(size=(1800*z, 900*z))
        ax = Axis(fig[1, 1], ylabel="Conflict", xlabel = "Time", xticklabelsvisible=false, yticklabelsvisible=false)

        for nreps in factors.repetitions
            for facts in factors.facts
                # Filter by pools and factors
                fdf = @chain df begin
                    subset(
                        :pool => ByRow(in([:Diet, :Hidden])),
                        :nreps => ByRow(==(nreps)),
                        :nfacts => ByRow(==(length(facts)))
                    )
                end

                # Uncertainty in the :Diet pool
                uncdf = @chain subset(fdf, :pool => ByRow(==(:Diet))) begin
                    groupby(:t)
                    combine(:act => uncertainty => :uncertainty)
                end

                # Familiarity in the :Hidden pool
                famdf = @chain subset(fdf, :pool => ByRow(==(:Hidden))) begin
                    groupby(:t)
                    combine(:act => familiarity => :familiarity)
                end

                # Plot conflict over time for each episode

                confdf = @chain innerjoin(uncdf, famdf, on=:t) begin
                    transform([:uncertainty, :familiarity] => ((u, f) -> u .* f) => :conflict)
                end
                lc = Makie.wong_colors()[2]
                lc = nreps == factors.repetitions.fewer ? Makie.wong_colors()[5] : lc
                lc = nreps == factors.repetitions.many ? Makie.wong_colors()[6] : lc
                ls = length(facts) == length(factors.facts.fewer) ? :dash : :solid
                lines!(ax, confdf.t, confdf.conflict, color=lc, linestyle=ls)
            end
        end

        # Legend
        c1 = LineElement(color=Makie.wong_colors()[2])
        c2 = LineElement(color=Makie.wong_colors()[5])
        c3 = LineElement(color=Makie.wong_colors()[6])
        l1 = LineElement(color=:black, linestyle=:dash)
        l2 = LineElement(color=:black, linestyle=:solid)
        Legend(fig[1, 2], [c3, c2, c1, l2, l1], ["$(factors.repetitions[3]) reps", "$(factors.repetitions[2]) reps", "$(factors.repetitions[1]) reps", "Many facts", "Few facts"])
        fig
    end
end

save("poster-panel3.svg", fig)