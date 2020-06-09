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
              plot_final_PICKS=True,
              plot_additional_PICKS={},
              show=False):
    """Comprehensive plotting function.

    This function will plot the input trace with all the necessary
    picking algorithm informations.

    hos_arr, eval_fun, hos_idx, pickTime_UTC must be HOST's dicts

    """

    trace = hostobj.tr
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
        fig = plt.figure()
        inax = fig.add_subplot(111)
    else:
        inax = plot_ax

    # Creating time vector and trace data
    tv = trace.times()
    td = trace.data
    if normalize:
        td = HS._normalize_trace(td, rangeVal=[-1, 1])

    # ============================ Create Colors

    my_color_list = ['sandybrown',
                     'deepskyblue',
                     'navy',
                     'darkorchid',
                     'lightseagreen',
                     'red',
                     'pink',
                     'grey',
                     'violet',
                     'brown',
                     'green',
                     'darkred',
                     'black',  # ]
                     # Extreme cases
                     'sandybrown',
                     'deepskyblue',
                     'navy',
                     'darkorchid',
                     'lightseagreen',
                     'red',
                     'pink',
                     'grey',
                     'violet',
                     'brown',
                     'green',
                     'darkred',
                     'black']

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
                                     mode='constant',
                                     constant_values=(np.nan,)) +
                                    (_ii+1)*shift_cf,
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle='-.',
                          label=_kk+" HOS")
            else:
                inax.plot(tv, np.pad(_aa, (zeropad_start, 0),
                                     mode='constant',
                                     constant_values=(np.nan,)),
                          color=my_color_list[_ii],
                          linewidth=1,
                          linestyle='-.',
                          label=_kk+" HOS")

        # EVAL:
        for _ii, (_kk, _aa) in enumerate(eval_fun.items()):
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
        for _kk, _pp in pickTime_UTC.items():
            if _kk not in ('mean', 'median'):
                inax.axvline(_pp - trace.stats.starttime,
                             color=my_color_list[col_idx],
                             linewidth=1.5,
                             label=_kk+" PICK")
                col_idx += 1

    # ============================ PICKS additional:
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

    # ============================ PICKS final:
    if plot_final_PICKS:
        inax.axvline(pickTime_UTC['mean'] - trace.stats.starttime,
                     color="gold",
                     linestyle="-",
                     linewidth=2.5,
                     label="mean PICK")
        inax.axvline(pickTime_UTC['median'] - trace.stats.starttime,
                     color="teal",
                     linestyle="-",
                     linewidth=3,
                     label="median PICK")

    # ============================ Plot TRACE
    inax.plot(tv, td, "k", label="trace")
    inax.set_xlabel("time (s)")
    inax.set_ylabel("counts")
    inax.legend(loc='lower left')
    inax.set_title(axtitle, {'fontsize': 16, 'fontweight': 'bold'})
    if show:
        plt.tight_layout()
        plt.show()
    #
    if not plot_ax:
        return (fig, inax)
    else:
        return inax
