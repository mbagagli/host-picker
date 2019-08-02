import numpy as np
from obspy.core.trace import Trace
import logging
# plot
import matplotlib.pyplot as plt
from host import scaffold as HS
from host import errors as HE
# from collections import OrderedDict
# from matplotlib.dates import AutoDateocator, AutoDateFormatter


logger = logging.getLogger(__name__)


def plot_HOST(trace,
              hos_arr,
              eval_fun,
              hos_idx,
              pickTime_UTC,
              normalize=True,
              plot_ax=None,
              show=False,
              axtitle="HOST picks"):
    """Comprehensive plotting function.

    This function will plot the input trace with all the necessary
    picking algorithm informations.

    hos_arr, eval_fun, hos_idx, pickTime_UTC must be HOST's dicts

    """
    #(hos_idx+N+1)
    if not isinstance(trace, Trace):
        raise HE.BadInstance({"message":
                              "Please input a valid obspy.core.Trace object"})
    if (not isinstance(hos_arr, dict) or
       not isinstance(eval_fun, dict) or
       not isinstance(hos_idx, dict) or
       not isinstance(pickTime_UTC, dict)):
        raise HE.BadInstance({"message":
                              "Positional parameter (apart from trace) " +
                              "must be a dict object"})
    if not plot_ax:
        inax = plt.axis()
    else:
        inax = plot_ax

    # Creating time vector and trace data
    tv = trace.times
    td = trace.data
    if normalize:
        td = HS.normalize_trace(td, rangeVal=[-1, 1])

    # Loop over dicts:
    for _kk in hos_arr.keys():
        # Plot HOS
        pass



    # Plot HOS arrays
    # StreamTrace + labels
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


def plot_host_stat(inarr, m, s, thresh, highlight=[]):
    """ For Gaussian mean/Std """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(inarr)
    ax1.axhline(y=m, color="grey", linestyle="--")
    ax1.axhline(y=thresh*s, color="gold", linestyle="--")
    if list(highlight):
        ax1.scatter(highlight, inarr[highlight], color='green')
    #
    plt.show()
    return fig, ax1


def plot_host_aic(st, hoscf, aicf, pidx):
    """ Debugging the MyAIC picker over the HOS_cf """
    QP.plot_QUAKE_CF(st,
                     {'KURT': hoscf,
                      'AIC': aicf},
                     chan=channel,
                     picks={'pick': pidx},
                     normalize=True,
                     show=True)


def plot_QUAKE_CF(st,
                  CF,
                  chan="*Z",
                  picks=None,
                  inax=None,
                  normalize=False,
                  axtitle='quake_cf',
                  shift_cf=None,
                  show=False):
    """
    Input is a obspy stream object
    Method to plot all the necessary CF of Quake pickers.

    CF is a dict containing the name of CF as key and np.array as data
    The key name will be used for plotting legend

    picks is a dict containing the name as key and idx/UTCDateTime as
    value. The key name will be used for plotting legend

    """
    if not isinstance(st, Stream):
        raise QE.InvalidVariable({"message":
                                  "Please input a valid obspy.stream object"})
    if not isinstance(CF, dict):
        raise QE.InvalidVariable({"message":
                                  "CF must be a dict containing the name of" +
                                  " CF as key and np.array as data"})
    if picks and not isinstance(picks, dict):
        raise QE.InvalidVariable({"message": "picks must be a dict"})
    #
    tr = st.select(channel=chan)[0]
    orig = tr.data
    df = tr.stats.sampling_rate
    # just transform in seconds the x ax (sample --> time)
    tv = np.array([ii / df for ii in range(len(orig))])
    #
    colorlst = ("r",
                "b",
                "g",
                "y",
                "c")
    # CFs
    if normalize:
        orig = normalizeTrace(orig, rangeVal=[-1, 1])

    for _ii, (_kk, _aa) in enumerate(CF.items()):
        if normalize:
            _aa = normalizeTrace(_aa, rangeVal=[0, 1])
        zeropad = len(orig) - len(_aa)
        if not inax:
            if shift_cf:
                plt.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                    constant_values=(np.nan,)) +
                             (_ii+1)*shift_cf,
                         colorlst[_ii], label=_kk)
            else:
                plt.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                    constant_values=(np.nan,)),
                         colorlst[_ii], label=_kk)
            if picks:
                for _pp, _tt in picks.items():
                    plt.axvline(picks[_pp]/df,
                                color=colorlst[_ii],
                                linewidth=2,
                                linestyle='dashed',
                                label=_pp)

        else:
            if shift_cf:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,))  +
                             (_ii+1)*shift_cf,
                          colorlst[_ii], label=_kk)
            else:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)),
                          colorlst[_ii], label=_kk)

            if picks:
                for _pp, _tt in picks.items():
                    inax.axvline(picks[_pp]/df,
                                 color=colorlst[_ii],
                                 linewidth=2,
                                 linestyle='dashed',
                                 label=_pp)

    # StreamTrace + labels
    if not inax:
        plt.plot(tv, orig, "k", label="trace")
        plt.xlabel("time (s)")
        plt.ylabel("counts")
        plt.legend(loc='lower left')
        plt.title(axtitle, {'fontsize': 16, 'fontweight': 'bold'}, loc='center')
    else:
        inax.plot(tv, orig, "k", label="trace")
        inax.set_xlabel("time (s)")
        inax.set_ylabel("counts")
        inax.legend(loc='lower left')
        inax.set_title(axtitle, {'fontsize': 16, 'fontweight': 'bold'})

    if show:
        plt.tight_layout()
        plt.show()
    return True
