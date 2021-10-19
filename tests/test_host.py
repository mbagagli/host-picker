from numpy import isclose, array, ndarray
from host.picker import Host
from obspy import read, UTCDateTime
from obspy.core.trace import Trace
#
from pprint import pprint


# ================================================   Preparing ...
# =====================================================================

def create_obspy_trace(data, stats):
    """ It will return an ObsPy.Trace instance from give array and
        stats dictionary
    """
    if not isinstance(data, ndarray):
        raise TypeError("DATA must be a numpy.ndarray instance!")
    if not isinstance(stats, dict):
        raise TypeError("STATS must be a dict instance!")
    #
    return Trace(data=data, header=stats)


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
trproc = stproc.select(channel="*Z")[0]


# ================================================ Starting the tests
# =====================================================================

def test_init():
    """ Test the picker init method """
    errors = []
    #
    try:
        HP = Host(trproc,
                  0.6,
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_f5': {'power': 2}},
                  detection_method="aic")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_setter():
    """ Test the setter """
    errors = []
    #
    try:
        HP = Host(trproc,
                  0.6,
                  hos_method="kurtosis",
                  detection_method="aic")
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method="aic")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 690000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 630000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 676667),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                     'outlier_obs': {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 630000)},
                     'pick_error': 0.029999999999999999,
                     'valid_obs': {'0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 690000),
                                   '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 680000),
                                   '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 660000)}}
    #
    if len(pickTime_UTC.keys()) != 10:
        errors.append("KEY numbers doesn't correspond, missing some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  0.7,
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method="aic")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'outlier_obs': {},
                     'pick_error': 3.19,
                     'valid_obs': {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000)}}
    #
    if len(pickTime_UTC.keys()) != 6:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method="aic")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 740000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 720000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                     'outlier_obs': {'0.15': UTCDateTime(2009, 8, 24, 0, 20, 7, 740000),
                                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 720000)},
                     'pick_error': 0.0,
                     'valid_obs': {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                                   '0.2': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
                                   '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000)}}
    #
    if len(pickTime_UTC.keys()) != 10:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  0.7,
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method="aic")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000),
                     'outlier_obs': {},
                     'pick_error': 3.19,
                     'valid_obs': {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 990000)}}
    #
    if len(pickTime_UTC.keys()) != 6:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  0.7,
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method=("diff", 0.5))
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 570000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 570000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 570000),
                     'outlier_obs': {},
                     'pick_error': 39.960000000000001,
                     'valid_obs': {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 570000)}}
    #
    if len(pickTime_UTC.keys()) != 6:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
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
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method=("diff", 0.5))
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 630000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 6, 740000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 610000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 620000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 6, 683333),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                     'outlier_obs': {'0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 610000),
                                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 620000)},
                     'pick_error': 0.11,
                     'valid_obs': {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 630000),
                                   '0.15': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                                   '0.2': UTCDateTime(2009, 8, 24, 0, 20, 6, 740000)}}
    #
    if len(pickTime_UTC.keys()) != 10:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_transform_cf():
    """Testing the new functionality of HOST picker.
       Select the transformation methods from list
    """
    errors = []
    #
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  [0.1, 0.15, 0.2, 0.25, 0.3],
                  hos_method="skewness",
                  transform_cf={'transform_f2': {},
                                'transform_smooth': {'window_type': 'hanning'}},
                  detection_method=("diff", 0.5))
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 630000),
                     '0.15': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                     '0.2': UTCDateTime(2009, 8, 24, 0, 20, 6, 740000),
                     '0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 610000),
                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 620000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 6, 683333),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                     'outlier_obs': {'0.25': UTCDateTime(2009, 8, 24, 0, 20, 7, 610000),
                                     '0.3': UTCDateTime(2009, 8, 24, 0, 20, 7, 620000)},
                     'pick_error': 0.11,
                     'valid_obs': {'0.1': UTCDateTime(2009, 8, 24, 0, 20, 6, 630000),
                                   '0.15': UTCDateTime(2009, 8, 24, 0, 20, 6, 680000),
                                   '0.2': UTCDateTime(2009, 8, 24, 0, 20, 6, 740000)}}
    #
    if len(pickTime_UTC.keys()) != 10:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_singlewin_kurt_min():
    """ Test the HOST picking algorithm for:
       - KURTOSIS
       - SINGLEWIN
       - MIN
    """
    errors = []
    #
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    try:
        HP = Host(tr,
                  0.7,
                  hos_method="kurtosis",
                  transform_cf={'transform_f2': {}, 'transform_f3': {}, 'transform_f4': {}},
                  detection_method="min")
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'outlier_obs': {},
                     'pick_error': 39.89,
                     'valid_obs': {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000)}}
    #
    if len(pickTime_UTC.keys()) != 6:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_work_singlewin_skew_s2nratio():
    """ Test the HOST picking algorithm for:
       - SKEWNESS
       - SINGLEWIN
       - AIC
    """
    errors = []
    #
    tr = trproc.copy()
    tr.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.250000"))
    #
    try:
        HP = Host(tr,
                  0.7,
                  hos_method="skewness",
                  detection_method='aic')
    except TypeError:
        errors.append("HOST class uncorrectly initialized")
    #
    HP.work(debug_plot=False)
    pickTime_UTC = HP.get_picks_UTC()
    ref_pick_dict = {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'mean': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'median': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000),
                     'outlier_obs': {},
                     'pick_error': 39.89,
                     'valid_obs': {'0.7': UTCDateTime(2009, 8, 24, 0, 20, 7, 700000)}}
    #
    if len(pickTime_UTC.keys()) != 6:
        errors.append("KEY numbers doesn't correspond, missins some")
    #
    for _kk, _pp in pickTime_UTC.items():
        if _kk != "pick_error" and pickTime_UTC[_kk] != ref_pick_dict[_kk]:
            errors.append("wrong KEY for pick %s" % _kk)
        elif _kk == "pick_error" and not isclose(pickTime_UTC[_kk], ref_pick_dict[_kk]):
            errors.append("wrong PICK-ERROR assessment of %5.3f" %
                          (pickTime_UTC[_kk] - ref_pick_dict[_kk]))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_only_s2nratio():
    """ Testing only the signal-to-noise ratio method """
    errors = []
    fake_data = array(
        [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.2, -0.1, 0.0,
         0.2, 0.5, 1, 2, 4, 6, 5, 4.5, 1.5])
    fake_stats = {'starttime': UTCDateTime("2009-08-24T00:20:00"),
                  'delta': 0.5}
    #
    tr = create_obspy_trace(fake_data, fake_stats)
    HP = Host(tr,
              0.7,
              hos_method="skewness",
              detection_method='aic')
    # 4 sample in noise, 4 sample in signal
    s2nr = HP._snratio(UTCDateTime("2009-08-24T00:20:06"), 2.0, 2.0)
    #
    if not isclose(s2nr, 6.10):
        errors.append("S2N errors doesn't correspond!")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
