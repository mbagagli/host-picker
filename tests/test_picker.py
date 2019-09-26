from host.picker import Host
from obspy import read, UTCDateTime
import pprint
import sys


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


straw = read('./tests_data/obspy_read.mseed')
stproc = miniproc(straw)

# ================================================ Starting the tests


def test_init():
    """ Test the picker init method """
    errors = []
    #
    try:
        HP = Host(stproc,
                  0.6,
                  channel="*Z",
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'windowtype': 'hanning'},
                                'transform_f5': {'power': 2}},
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


def test_work_multiwin_kurt_aic():
    """ Test the HOST picking algorithm for:
       - KURTOSIS
       - MULTIWIN
       - AIC
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  channel="*Z",
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 640000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 590000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 668000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000)}
    #
    if len(pickTime_UTC.keys()) != 7:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_singlewin_kurt_aic():
    """ Test the HOST picking algorithm for:
       - KURTOSIS
       - SINGLEWIN
       - AIC
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  0.7,
                  channel="*Z",
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 8, 100000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 8, 100000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 8, 100000)}
    #
    if len(pickTime_UTC.keys()) != 3:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s", _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_multiwin_skew_aic():
    """ Test the HOST picking algorithm for:
       - SKEWNESS
       - MULTIWIN
       - AIC
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  channel="*Z",
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 810000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 840000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 860000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 890000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 910000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 862000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 860000)}
    #
    if len(pickTime_UTC.keys()) != 7:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s", _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_singlewin_skew_aic():
    """ Test the HOST picking algorithm for:
       - SKEWNESS
       - SINGLEWIN
       - AIC
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  0.7,
                  channel="*Z",
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="aic",
                  diff_gauss_thresh=None)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 8, 110000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 8, 110000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 8, 110000)}

    #
    if len(pickTime_UTC.keys()) != 3:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s", _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_singlewin_skew_diff():
    """ Test the HOST picking algorithm for:
       - SKEWNESS
       - SINGLEWIN
       - DIFFERENCE
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  0.7,
                  channel="*Z",
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="diff",
                  diff_gauss_thresh=0.5)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 500000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 500000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 500000)}
    #
    if len(pickTime_UTC.keys()) != 3:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_multiwin_skew_diff():
    """ Test the HOST picking algorithm for:
       - SKEWNESS
       - MULTIWIN
       - DIFFERENCE
    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  channel="*Z",
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="diff",
                  diff_gauss_thresh=0.5)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 650000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 600000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 458000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000)}
    #
    if len(pickTime_UTC.keys()) != 7:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_transform_cf():
    """Testing the new functionality of HOST picker.
       Select the transformation methods from list

    """
    errors = []
    #
    st = stproc.copy()
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(st,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  channel="*Z",
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_f4': {'window_type': 'hanning'}},
                  detection_method="diff",
                  diff_gauss_thresh=0.5)
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 650000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 600000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 458000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000)}
    #
    if len(pickTime_UTC.keys()) != 7:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
