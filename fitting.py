import numpy
import pandas
import conntility

from distribution import cut_zeros_and_shift, sis

__args = ["A_i", "B_i", "A_f", "B_f", "C_f", "p"]
__reasonable_initial_guess = {
    "A_i": 1.0,
    "B_i": 0.005,
    "A_f": 0.1,
    "B_f": 0.002,
    "C_f": 0.5,
    "p": 0.5
}
__parameter_bounds = {
    "A_i": (0.01, 1.0),
    "B_i": (1E-6, 0.1),
    "A_f": (0.0, 1.0),
    "B_f": (1E-9, 0.1),
    "C_f": (0.0, 1.0),
    "p": (0.0, 1.0)
}

def _build_initial_guess(arg_lo):
    return [
        __reasonable_initial_guess[_str]
        for _str, _idx in zip(__args, arg_lo)
        if _idx >= 0
    ]

def _build_bounds(arg_lo):
    return [
        __parameter_bounds[_str]
        for _str, _idx in zip(__args, arg_lo)
        if _idx >= 0
    ]

def build_cost_function(edge_table, bin_centers, max_num_touches=100, **kwargs):
    mean_extra_edges = edge_table.groupby("bin")["count"].mean().values - 1
    var_extra_edges = edge_table.groupby("bin")["count"].var().values
    mask = (mean_extra_edges > 0) & (var_extra_edges > 0)

    idx = 0
    arg_lo = []
    for _arg in __args:
        if _arg in kwargs:
            arg_lo.append(-1)
        else:
            arg_lo.append(idx)
            idx = idx + 1
    
    def out_fun(args, plot=False):
        A_i, B_i, A_f, B_f, C_f, p = [
            args[_i] if _i >= 0 else kwargs[_str]
            for _i, _str in zip(arg_lo, __args)
        ]
        i = numpy.minimum(1.0, A_i * numpy.exp(-bin_centers * B_i))
        f = numpy.minimum(1.0-1E-6, A_f * numpy.exp(-bin_centers * B_f) + C_f)
        base_distr, _ = cut_zeros_and_shift(sis(i, f, p, max_num_touches))
        mn, var = base_distr.stats()

        if plot:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)

            ax1.plot(bin_centers, numpy.vstack([mn, var]).transpose())

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(bin_centers, mean_extra_edges)
            ax2.plot(bin_centers, var_extra_edges)
            ax2.set_ylim(ax1.get_ylim())
        return (numpy.abs(mean_extra_edges[mask] - mn[mask]) / mean_extra_edges[mask]).sum() +\
           (numpy.abs(var_extra_edges[mask] - var[mask]) / var_extra_edges[mask]).sum()
    print("Number of args to optimize: {0}".format(numpy.sum([_x >= 0 for _x in arg_lo])))
    return out_fun, _build_initial_guess(arg_lo), _build_bounds(arg_lo), arg_lo

def optimize_touch_model(edge_table, bin_centers, max_num_touches=100, **kwargs):
    from scipy.optimize import minimize
    opt_fun, initial_guess, bounds, arg_lo = build_cost_function(edge_table, bin_centers, max_num_touches=max_num_touches,
                                                         **kwargs)
    sol = minimize(opt_fun, initial_guess, bounds=bounds)
    print(sol.message)
    opt_params = dict([
                      (_str, sol.x[_i]) if _i >= 0 else (_str, kwargs[_str])
                       for _i, _str in zip(arg_lo, __args)
                       ])
    opt_models = {
        "i": numpy.minimum(1.0, opt_params["A_i"] * numpy.exp(-bin_centers * opt_params["B_i"])),
        "f": numpy.minimum(1.0-1E-6, opt_params["A_f"] * numpy.exp(-bin_centers * opt_params["B_f"]) + opt_params["C_f"]),
        "p": opt_params["p"]
    }
    return opt_params, opt_models

def keep_edges_randomized_touch_count(M, fitted_parameters, max_touch_count, bin_prop_name="bin"):
    bin_ids = M.edges[bin_prop_name].values
    u_bin_ids = numpy.unique(bin_ids)
    rnd_touches = numpy.zeros(len(M.edges), dtype=int)
    for bin_id in u_bin_ids:
        mask = bin_ids == bin_id
        mdl_extra_touches, _ = cut_zeros_and_shift(  # Convert to distribution of extra touches
                            sis(
                                fitted_parameters["i"][bin_id],
                                fitted_parameters["f"][bin_id],
                                fitted_parameters["p"],
                                max_touch_count))
        rnd_touches[mask] = mdl_extra_touches.rvs(size=mask.sum()) + 1
    edge_df = pandas.DataFrame({
        "count": rnd_touches,
        bin_prop_name: bin_ids
    })
    return conntility.ConnectivityMatrix(M._edges, vertex_properties=M._vertex_properties,
                                        edge_properties=edge_df,
                                        shape=M._shape, default_edge_property="count")


def randomize_edges_and_count(M, fitted_parameters, max_touch_count, distance_bins, distance_cols=["x", "y", "z"]):
    from scipy.spatial import distance
    distance_cols=["x", "y", "z"]
    D = distance.squareform(distance.pdist(M.vertices[distance_cols]))
    D = numpy.digitize(D, bins=distance_bins) - 1
    edge_row = []
    edge_col = []
    edge_count = []
    edge_bin_id = []

    for bin_id in range(len(distance_bins)):
        print(bin_id)
        nz_row, nz_col = numpy.nonzero(D == bin_id)
        mdl = sis(fitted_parameters["i"][bin_id],
                  fitted_parameters["f"][bin_id],
                  fitted_parameters["p"],
                  max_touch_count)
        rvs = mdl.rvs(size=len(nz_row))
        edge_row.extend(nz_row[rvs > 0])
        edge_col.extend(nz_col[rvs > 0])
        edge_count.extend(rvs[rvs > 0])
        edge_bin_id.extend(numpy.ones(len(edge_count) - len(edge_bin_id), dtype=int) * bin_id)

    edge_idx = pandas.DataFrame({
        "row": numpy.array(edge_row),
        "col": numpy.array(edge_col)
    })
    edge_df = pandas.DataFrame({
            "count": numpy.array(edge_count),
            "bin": numpy.array(edge_bin_id)
        })
    M_random = conntility.ConnectivityMatrix(edge_idx, vertex_properties=M._vertex_properties,
                                             edge_properties=edge_df,
                                             shape=M._shape, default_edge_property="count")
    return M_random
