from host.picker import Host
import host.errors as HE
from obspy import read, UTCDateTime
import numpy as np


def miniproc(st):
    prs = st.copy()
    prs.detrend('demean')
    prs.detrend('simple')
    prs.taper(max_percentage=0.05, type='cosine')
    prs.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return prs


straw = read()
stproc = miniproc(straw)

# ================================================ Starting the tests


def test_init():
    """ Test the picker """
    errors = []
    #
    try:
        HP = Host(stproc,
                  0.6,
                  channel="*Z",
                  hos_method="kurtosis",
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_setter():
    """ Test the setter """
    errors = []
    #
    try:
        HP = Host(stproc,
                  0.6,
                  channel="*Z",
                  hos_method="kurtosis",
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    if not isinstance(HP.time_win, tuple):
        errors.append("single time window not stored")
    if HP.method != "kurt":
        errors.append("hos_method not stored")
    if HP.detection != "aic":
        errors.append("detection_method not stored")
    if HP.thresh:
        errors.append("diff_gauss_thresh erroneusly stored")
    #
    HP.set_time_windows([0.1, 0.2, 0.3])
    if not isinstance(HP.time_win, list):
        errors.append("set_time_windows ERROR")
    #
    HP.set_hos_method("skewness")
    if HP.method != "skew":
        errors.append("set_hos_method ERROR")
    #
    HP.set_detection_method("gauss")
    if HP.detection != "diff":
        errors.append("set_detection_method ERROR")
    #
    HP.set_diffgauss_threshold(2.5)
    if HP.thresh != 2.5:
        errors.append("set_diffgauss_threshold ERROR")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work():
    """ Test the setter """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    st.plot()
    try:
        HP = Host(st,
                  0.6,
                  channel="*Z",
                  hos_method="kurtosis",
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
