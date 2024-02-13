module RandMirror

include(joinpath("utilities", "classification_metrics.jl"))
include(joinpath("utilities", "data_generation.jl"))
include(joinpath("utilities", "multiple_plots.jl"))

include(joinpath("utilities", "variable_selection_plus_inference.jl"))
include(joinpath("utilities", "mirror_statistic.jl"))
include(joinpath("utilities", "randomisation_ds.jl"))

include(joinpath("utilities", "wrapper_pipeline_inference.jl"))
include(joinpath("utilities", "simulation_runner.jl"))

end # module RandMirror
