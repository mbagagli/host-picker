import numpy as np
from obspy.core.trace import Trace
import logging
# plot
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
#
from host import scaffold as HS
from host import errors as HE
# from collections import OrderedDict
# from matplotlib.dates import AutoDateocator, AutoDateFormatter


logger = logging.getLogger(__name__)


def plot_HOST(trace,
              hos_arr,
              eval_fun,
              pickTime_UTC,
              normalize=True,
              plot_ax=None,
              axtitle="HOST picks",
              shift_cf=False,
              plot_HOS=False,
              plot_EVAL=False,
              plot_intermediate_PICKS=False,
              plot_final_PICKS=True,
              show=False):
    """Comprehensive plotting function.

    This function will plot the input trace with all the necessary
    picking algorithm informations.

    hos_arr, eval_fun, hos_idx, pickTime_UTC must be HOST's dicts

    """
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
        inax = plt.axes()
    else:
        inax = plot_ax

    # Creating time vector and trace data
    tv = trace.times()
    td = trace.data
    if normalize:
        td = HS._normalize_trace(td, rangeVal=[-1, 1])

    # -------------------------- Create Colors

    my_color_list = ['sandybrown',
                     'deepskyblue',
                     'navy',
                     'darkorchid',
                     'lightseagreen',
                     'coral',
                     'orange']

    # -------------------------- Loop over dicts
    # HOS:
    if plot_HOS:
        for _ii, (_kk, _aa) in enumerate(hos_arr.items()):
            if normalize:
                _aa = HS._normalize_trace(_aa, rangeVal=[0, 1])
            zeropad = len(td) - len(_aa)
            #
            if shift_cf:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle='-.',
                          label=_kk+" HOS")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle='-.',
                          label=_kk+" HOS")

    # EVAL:
    if plot_EVAL:
        for _ii, (_kk, _aa) in enumerate(eval_fun.items()):
            if normalize:
                _aa = HS._normalize_trace(_aa, rangeVal=[0, 1])
            zeropad = len(td) - len(_aa)
            #
            if shift_cf:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=':',
                          label=_kk+" EVAL")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=':',
                          label=_kk+" EVAL")

    # PICKS:
    for _ii, (_kk, _pp) in enumerate(pickTime_UTC.items()):
        if _kk == "median":
            if plot_final_PICKS:
                inax.axvline(_pp - trace.stats.starttime,
                             color="r",
                             linestyle="-",
                             linewidth=2,
                             label=_kk+" PICK")
        elif _kk == "mean":
            if plot_final_PICKS:
                inax.axvline(_pp - trace.stats.starttime,
                             color="gold",
                             linewidth=2,
                             label=_kk+" PICK")
        else:
            if plot_intermediate_PICKS:
                inax.axvline(_pp - trace.stats.starttime,
                             color=my_color_list[_ii],
                             linewidth=1.5,
                             label=_kk+" PICK")

    # -------------------------- Plot TRACE
    inax.plot(tv, td, "k", label="trace")
    inax.set_xlabel("time (s)")
    inax.set_ylabel("counts")
    inax.legend(loc='lower left')
    inax.set_title(axtitle, {'fontsize': 16, 'fontweight': 'bold'})
    if show:
        plt.tight_layout()
        plt.show()
    #
    return inax


# ================== TIPS
# To create a continuos discrete list of colors with matplotlib

#    NUM_COLORS = len(pickTime_UTC.keys()) - 2  # MB: - mean and  -median
#    cm = plt.get_cmap('gist_rainbow')
#    cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
#    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
#    inax.set_prop_cycle(color=[scalarMap.to_rgba(i)
#                               for i in range(NUM_COLORS)])
