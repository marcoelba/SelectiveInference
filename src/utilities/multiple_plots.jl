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


function grouped_boxplot(;df, group_var, var_columns, title_plot, sub_titles)

    max_y = maximum(maximum(eachrow(df[!, var_columns])))
    min_y = minimum(minimum(eachrow(df[!, var_columns])))
    n_plots = length(var_columns)

    l = @layout [grid(1, n_plots)]

    # plot 1
    p1 = @df df boxplot(string.(cols(group_var)), cols(Symbol(var_columns[1])), group=cols(group_var), label=false)
    ylims!((min_y, max_y))
    xlabel!(string(group_var))
    title!(sub_titles[1])

    all_plots = [p1]

    for (ii, var_i) in enumerate(var_columns[2:n_plots])
        p_l = @df df boxplot(string.(cols(group_var)), cols(Symbol(var_i)), group=cols(group_var), label=false)
        ylims!((min_y, max_y))
        xlabel!(string(group_var))
        title!(sub_titles[ii])

        push!(all_plots, p_l)
        # plot!(all_plots, p_l)
    end
    
    plot_all = plot(all_plots..., layout = l)
    plot_all[:plot_title] = title_plot
    plot(plot_all)
end
