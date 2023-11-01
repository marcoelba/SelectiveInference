# Plots

using StatsPlots
using Plots
using DataFrames
using Statistics


function triple_boxplot(;df, col_names, title_plot)
    max_y = maximum(
        (maximum(df[!, col_names[1]]),
        maximum(df[!, col_names[2]]),
        maximum(df[!, col_names[3]]))
    )
    min_y = minimum(
        (minimum(df[!, col_names[1]]),
        minimum(df[!, col_names[2]]),
        minimum(df[!, col_names[3]]))
    )
    l = @layout [grid(1, 3)]
    # plot 1
    p1 = boxplot(df[!, col_names[1]], label=false, xaxis=false)
    ylims!((min_y, max_y))
    xlabel!(string(col_names[1]))
    # plot 2
    p2 = boxplot(df[!, col_names[2]], label=false, xaxis=false)
    ylims!((min_y, max_y))
    xlabel!(string(col_names[2]))
    # plot 3
    p3 = boxplot(df[!, col_names[3]], label=false, xaxis=false)
    ylims!((min_y, max_y))
    xlabel!(string(col_names[3]))

    all_p = plot(p1, p2, p3, layout = l)
    all_p[:plot_title] = title_plot
    plot(all_p)
end


function grouped_boxplot(;df, group_var, var_columns, title_plot, sub_titles, sub_titlesize=10, titlesize=15)

    max_y = maximum(maximum(eachrow(df[!, var_columns])))
    min_y = minimum(minimum(eachrow(df[!, var_columns])))
    n_plots = length(var_columns)

    l = @layout [grid(1, n_plots)]

    # plot 1
    p1 = @df df boxplot(string.(cols(group_var)), cols(Symbol(var_columns[1])), group=cols(group_var), label=false)
    ylims!((min_y, max_y))
    xlabel!(string(group_var))
    title!(sub_titles[1], titlefontsize=sub_titlesize)

    all_plots = [p1]

    for plot_i in range(2, n_plots)
        p_l = @df df boxplot(string.(cols(group_var)), cols(Symbol(var_columns[plot_i])), group=cols(group_var), label=false)
        ylims!((min_y, max_y))
        xlabel!(string(group_var))
        title!(sub_titles[plot_i], titlefontsize=sub_titlesize)

        push!(all_plots, p_l)
    end
    
    plot_all = plot(all_plots..., layout = l, plot_titlevspan=0.1)
    plot_all[:plot_title] = title_plot
    plot(plot_all)
end


function plot_all(;df, var_columns, methods_to_evaluate)

    if var_columns == "fdr"
        var_columns = ["FDR_" * method for method in methods_to_evaluate]
        title_metric = "False Discovery Rate"
    elseif var_columns == "tpr"
        var_columns = ["TPR_" * method for method in methods_to_evaluate]
        title_metric = "True Positive Rate"
    end

    unique_rho = unique(df[:, "rho"])
    unique_beta = unique(df[:, "signal_strength"])

    if length(unique_beta) > 1
        for rho in unique_rho
            gp = grouped_boxplot(
                df=df[df[:, "rho"] .== rho, :],
                group_var=:signal_strength,
                var_columns=var_columns,
                title_plot="$(title_metric) - rho=$(rho)",
                sub_titles=methods_to_evaluate,
                sub_titlesize=10.
            )
            display(gp)
        end
    end

    if length(unique_rho) > 1
        for beta in unique_beta
            gp = grouped_boxplot(
                df=df[df[:, "signal_strength"] .== beta, :],
                group_var=:rho,
                var_columns=var_columns,
                title_plot="$(title_metric) - signal strength=$(beta)",
                sub_titles=methods_to_evaluate,
                sub_titlesize=10.
            )
            display(gp)
        end
    end

end
