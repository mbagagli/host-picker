import numpy as np
from obspy.core.trace import Trace
import logging
import matplotlib.pyplot as plt
#
from host import scaffold as HS
from host import errors as HE


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
              plot_additional_PICKS={},
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
                     'red',
                     'pink',
                     'grey',
                     'violet',
                     'brown']

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
            # GoingToC: replace INFs/NANs at the start and end with
            #           adiacent values for plotting reasons.
            #           This is introduced mainly for AIC EVAL.
            #           It will not affect GAUSSIAN-EVAL as well
            zeropad_start = td.size - _aa.size - 1
            _aa[0] = _aa[1]
            _aa[-1] = _aa[-2]
            _tt = _aa[~np.isnan(_aa)]
            _tt = _tt[~np.isinf(_aa)]
            zeropad_end = _aa.size - _tt.size + 1
            if normalize:
                _aa = HS._normalize_trace(_tt, rangeVal=[0, 1])
            #
            if shift_cf:
                inax.plot(tv, np.pad(_aa, (zeropad_start, zeropad_end), mode='constant',
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=':',
                          label=_kk+" EVAL")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad_start, zeropad_end), mode='constant',
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle=':',
                          label=_kk+" EVAL")

    # PICKS intermediate:
    col_idx = 0
    if plot_intermediate_PICKS:
        for _kk, _pp in pickTime_UTC.items():
            if _kk not in ('mean', 'median'):
                inax.axvline(_pp - trace.stats.starttime,
                             color=my_color_list[col_idx],
                             linewidth=1.5,
                             label=_kk+" PICK")
                col_idx += 1

    # PICKS additional:
    my_color_list_add = ['lime',
                         'forestgreen',
                         'limegreen',
                         'darkgreen']
    col_idx = 0
    if plot_additional_PICKS and isinstance(plot_additional_PICKS, dict):
        for _kk, _pp in plot_additional_PICKS.items():
            inax.axvline(_pp - trace.stats.starttime,
                         color=my_color_list_add[col_idx],
                         linewidth=1.5,
                         label=_kk)
            col_idx += 1

    # PICKS final:
    if plot_final_PICKS:
        inax.axvline(pickTime_UTC['mean'] - trace.stats.starttime,
                     color="gold",
                     linestyle="-",
                     linewidth=2,
                     label="mean PICK")
        inax.axvline(pickTime_UTC['median'] - trace.stats.starttime,
                     color="teal",
                     linestyle="-",
                     linewidth=2,
                     label="median PICK")

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
