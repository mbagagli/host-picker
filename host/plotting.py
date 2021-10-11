import numpy as np
from obspy.core.trace import Trace
import logging
import matplotlib.pyplot as plt
#
from host import scaffold as HS
from host import errors as HE


logger = logging.getLogger(__name__)


def plot_HOST(hostobj,
              normalize=True,
              plot_ax=None,
              axtitle="HOST picks",
              shift_cf=False,
              debug_plot=False,
              plot_final_picks=True,
              plot_additional_picks={},
              show=False):
    """ Comprehensive plotting function for HOST object.

    This function will plot the input trace with all the functions
    used to define the picks: CF and detection function (e.g., AIC if
    present).

    Args:
        hostobj (host.Host): HOST object class

    Optional:
        normalize (bool): if `True` it will normalize the main trace
            between -1 to 1
        plot_ax (matplotlib.axes.Axes): if given, the axes plot will be
            overridden
        axtitle (:obj:`str`): selfexplanatory, axis title
        shift_cf (:obj:`bool`): shift the CF in the x-axis
        debug_plot (:obj:`bool`): if True, all CF and eval functions
            from the host-windows used will be shown
        plot_final_picks (:obj:`bool`): if True will plot the final pick
        plot_additional_picks (:obj:`bool`): if True will plot the
            individual picks of each window. Useful when multi-windows
            option is used.
        show (:obj:`bool`): if True , figure will pop-up.

    Returns:
        fig (matplotlib.figure.Figure): the figure handle where the
            plot is made
        ax (matplotlib.axes.Axes): the axis handle
            where the plot is made

    """

    trace = hostobj.tr.copy()
    hos_arr = hostobj.hos_arr
    eval_fun = hostobj.eval_fun
    pickTime_UTC = hostobj.pickTime_UTC
    detection = hostobj.detection     # gauss / minima / aic

    if not isinstance(trace, Trace):
        raise HE.BadInstance({"message":
                              "Please input a valid obspy.core.Trace object"})
    if (not isinstance(hos_arr, dict) or
       not isinstance(eval_fun, dict) or
       not isinstance(pickTime_UTC, dict)):
        raise HE.BadInstance({"message":
                              "Positional parameter (apart from trace) " +
                              "must be a dict object"})

    if not plot_ax:
        fig = plt.figure(figsize=[10.5, 3.3])  # in inches
        inax = fig.add_subplot(111)
    else:
        inax = plot_ax

    # Creating time vector and trace data
    trace_start_time = trace.stats.starttime
    tv = trace.times()
    td = trace.data
    if normalize:
        td = HS._normalize_trace(td, rangeVal=[-1, 1])

    # ============================ Create Colors and define boundaries

    non_pick_related_keys =("mean",
                            "median",
                            "valid_obs",
                            "outlier_obs",
                            "pick_error")

    my_color_list = ["sandybrown",
                     "deepskyblue",
                     "navy",
                     "darkorchid",
                     "lightseagreen",
                     "red",
                     "pink",
                     "grey",
                     "violet",
                     "brown",
                     "green",
                     "darkred",
                     "black",  # ]
                     # Extreme cases
                     "sandybrown",
                     "deepskyblue",
                     "navy",
                     "darkorchid",
                     "lightseagreen",
                     "red",
                     "pink",
                     "grey",
                     "violet",
                     "brown",
                     "green",
                     "darkred",
                     "black"]

    # ============================ Pick-Error-Band
    if len(pickTime_UTC['valid_obs']) >= 2:
        _tt = tuple(pickTime_UTC['valid_obs'].values())
        inax.axvspan(_tt[0]-trace_start_time,
                     _tt[-1]-trace_start_time,
                     alpha=0.6, color='darkgray')

    # ============================ Loop over dicts (DEBUG)

    if debug_plot:
        # HOS:
        for _ii, (_kk, _aa) in enumerate(hos_arr.items()):
            if normalize:
                _aa = HS._normalize_trace(_aa, rangeVal=[0, 1])
            zeropad_start = tv.size - _aa.size
            #
            if shift_cf and isinstance(shift_cf, (int, float)):
                inax.plot(tv, np.pad(_aa, (zeropad_start, 0),
                                     mode="constant",
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle="-.",
                          label=_kk+" HOS")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad_start, 0),
                                     mode="constant",
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle="-.",
                          label=_kk+" HOS")

        # EVAL:
        for _ii, (_kk, _aa) in enumerate(eval_fun.items()):
            if detection.lower() == "minima":
                continue
            # GoingToC: replace INFs/NANs at the start and end with
            #           adiacent values for plotting reasons.
            #           This is introduced mainly for AIC EVAL.
            #           It will not affect GAUSSIAN-EVAL as well
            if detection.lower() == "aic":
                N = _aa.size
                zeropad_start = tv.size - N - 1
                _zpl = 0
                for _dd in range(0, _aa.size):
                    if np.isnan(_aa[_dd]) or np.isinf(_aa[_dd]):
                        _zpl += 1
                    else:
                        break

                _aa = _aa[~np.isnan(_aa)]
                _aa = _aa[~np.isinf(_aa)]
                Nnew = _aa.size

                zeropad_start = zeropad_start + _zpl
                zeropad_end = tv.size - Nnew - zeropad_start
                if normalize:
                    _aa = HS._normalize_trace(_aa, rangeVal=[0, 1])
            else:
                zeropad_start = td.size - _aa.size
                zeropad_end = 0
                if normalize:
                    _aa = HS._normalize_trace(_aa, rangeVal=[0, 1])
            #
            if shift_cf and isinstance(shift_cf, (int, float)):
                inax.plot(tv, np.pad(_aa, (zeropad_start, zeropad_end), mode="constant",
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=":",
                          label=_kk+" EVAL")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad_start, zeropad_end), mode="constant",
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=":",
                          label=_kk+" EVAL")

        # PICKS intermediate:
        col_idx = 0
        for _kk, _pp in pickTime_UTC.items():
            if _kk not in non_pick_related_keys:
                inax.axvline(_pp - trace.stats.starttime,
                             color=my_color_list[col_idx],
                             linewidth=1.5,
                             label=_kk+" PICK")
                col_idx += 1

    # ============================ PICKS additional:
    my_color_list_add = ["lime",
                         "forestgreen",
                         "limegreen",
                         "darkgreen"]
    col_idx = 0
    if plot_additional_picks and isinstance(plot_additional_picks, dict):
        for _kk, _pp in plot_additional_picks.items():
            inax.axvline(_pp - trace.stats.starttime,
                         color=my_color_list_add[col_idx],
                         linewidth=1.5,
                         label=_kk)
            col_idx += 1

    # ============================ PICKS final:
    if plot_final_picks:
        inax.axvline(pickTime_UTC["mean"] - trace.stats.starttime,
                     color="gold",
                     linestyle="-",
                     linewidth=2.5,
                     label="mean PICK")
        inax.axvline(pickTime_UTC["median"] - trace.stats.starttime,
                     color="teal",
                     linestyle="-",
                     linewidth=3,
                     label="median PICK")

    # ============================ Plot TRACE
    inax.plot(tv, td, "k", label="trace")
    inax.set_xlabel("time (s)")
    inax.set_ylabel("counts")
    inax.legend(loc="lower left")
    inax.set_title(axtitle, {"fontsize": 16, "fontweight": "bold"})
    if show:
        plt.tight_layout()
        plt.show()
    #
    if not plot_ax:
        return (fig, inax)
    else:
        return inax
