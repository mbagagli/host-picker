import logging
import numpy as np
#
from obspy.core import Stream
from obspy.core import UTCDateTime
#
from host import errors as HE
from host import plotting as HP
from host import scaffold as HS

logger = logging.getLogger(__name__)


"""Main picker module of HOST picking algorithm

In this module are stored all the functions and classes needed to pick
with HOS algorithms (skewness/kurtosis).

For a detailed explanation of the usage, the user should look the docs.

"""


class Host(object):
    """Main class for high horder statistics picker

    This class contains all the necessary parameter to run the picker
    and defining the obtained pick

    Note:
        The !WEIRD! error message are raised when the __init__ method
        fails on detecting erroneous input parameter. In fact these
        WEIRD error are raised from internal functions after some
        IF switches.

    """
    def __init__(self,
                 stream,
                 time_windows,
                 channel="*Z",
                 hos_method="kurtosis",
                 detection_method="aic",
                 diff_gauss_thresh=None):

        # stream checks
        if isinstance(stream, Stream):
            self.st = stream
            self.tr = stream.select(channel=channel)[0]
            self.dt = self.tr.stats.delta
            self.ts = self.tr.data
        else:
            logger.error("Not a valid ObsPy Stream instance")
            raise HE.BadInstance()

        # sliding windows checks
        if isinstance(time_windows, (float, int)):
            self.time_win = (time_windows,)
        elif isinstance(time_windows, (list, tuple)):
            self.time_win = time_windows
        else:
            logger.error("Input time windows is not a valid type")
            raise HE.BadInstance()

        # methods checks
        if hos_method.lower() in ('kurtosis', 'kurt', 'k'):
            self.method = "kurt"
        elif hos_method.lower() in ('skewness', 'skew', 's'):
            self.method = "skew"
        else:
            logger.error("HOS method not valid ['kurtosis'/'skewness']")
            raise HE.BadParameterValue()

        # detection method checks
        if detection_method.lower() in ('aic', 'akaike', 'a'):
            self.detection = "aic"
            self.thresh = None
        elif detection_method.lower() in ('diff', 'gauss'):
            self.detection = "diff"
            self.thresh = diff_gauss_thresh
        else:
            logger.error("DETECTION method Not valid ['aic'; 'diff'/'gauss']")
            raise HE.BadParameterValue()

        # Initialize output
        self.pickTime_UTC = {}
        self.hos_arr = {}
        self.eval_fun = {}
        self.hos_idx = {}

    # ============================== Private methods
    def _preprocess(self):
        """Minimal precprocessing method:
          - remove mean
          - remove linear trend
        """
        self.tr.detrend('demean')
        self.tr.detrend('simple')

    def _calculate_CF(self, tw):
        if self.ts.size == 0:
            logger.error("Missing trace time series data ... abort")
            raise HE.MissingAttribute()
        if tw <= self.dt:
            logger.error(("TimeWindow must be greater than trace's delta." +
                          " IN: %f - DELTA: %f ") % (float(tw),
                                                     float(self.dt)))
            raise HE.BadParameterValue()
        # --- Calculate CF
        logger.info("Calculating CF with HOS: %s" % self.method.upper())
        hos_arr = np.array([])
        N = round(tw/self.dt) + 1    # (16.16)
        for j, x in enumerate(self.ts):
            if j >= N:  # if j >= N: #new  j >=1 #old
                mk2 = HS.sixteen_eightteen(self.ts, j, N=N, k=2)     # 16.18
                if self.method == "kurt":
                    mk4 = HS.sixteen_eightteen(self.ts, j, N=N, k=4)
                    hos_arr = np.append(hos_arr, mk4/(mk2**2))
                elif self.method == "skew":
                    mk3 = HS.sixteen_eightteen(self.ts, j, N=N, k=3)
                    hos_arr = np.append(hos_arr, mk3/(mk2**3/2))
                else:
                    logger.error("!WEIRD! Wrong HOS method given " +
                                 "['skew'/'kurt']")
                    raise HE.BadParameterValue()
        #
        return hos_arr, N

    def _detect_pick(self, hos_arr):
        """Use one of the possible method to detect the pick over an
           HOS carachteristic function.

        """
        logger.info("Detecting PICK with %s" % self.detection.upper())
        if self.detection.lower() in ('diff', 'gauss'):
            if not self.thresh:
                logger.error("Missing threshold for 'diff' DETECTION method")
                raise HE.MissingAttribute()
            hos_idx, m, s, all_idx, eval_fun = HS.gauss_dev(hos_arr,
                                                            self.thresh)

        elif self.detection.lower() in ('aic', 'akaike'):
            hos_idx, eval_fun = HS.AICcf(hos_arr)

        else:
            logger.error("!WEIRD! Invalid pick extr. mode: %s " +
                         "['aic'; 'diff'/'gauss']" % self.detection)
            raise HE.BadParameterValue()
        #
        return hos_idx, eval_fun

    # ============================== Public methods

    def pick(self, tw):
        """ This method will calculate first the CF, and then
            detect the pick.

        """
        self._preprocess()
        # --- Calculate CF
        hos_arr, N = self._calculate_CF(tw)

        # MB: Create absolute value to ease the work of self_detect_pick
        hos_arr = HS.transform_f2(hos_arr)

        # MB: remove linear trend
        # hos_arr = HS.transform_f3(hos_arr)

        # MB: smooth HOS_CF (next line)
        hos_arr = HS.transform_f4(hos_arr, N, window_type='hanning')

        # --- Extract Pick (AIC/GAUSS)
        hos_idx, eval_fun = self._detect_pick(hos_arr)

        # --- Closing
        # time= NUMsamples/df OR NUMsamples*dt
        logger.debug("HOS: %s - PICKSEL: %s - idx: %r" % (self.method,
                                                          self.detection,
                                                          hos_idx+N+1))
        pickTime_UTC = self.tr.stats.starttime + ((hos_idx+N+1) * self.dt)
        return float(pickTime_UTC), hos_arr, eval_fun, (hos_idx+N+1)

    def work(self, debug_plot=False):
        """Main public method to run the picking algotrithm

        If the  time_win attributes is a set of values the output will
        be given in a dictionary containing as keys the diferent
        time windows values

        """
        # Check time windows
        if isinstance(self.time_win, (list, tuple)):
            _pt_UTC = {}
            _pt_float = {}
            _hos = {}
            _eval = {}
            _hos_idx = {}
            #
            for tw in self.time_win:
                (_pt_float[str(tw)],
                 _hos[str(tw)],
                 _eval[str(tw)],
                 _hos_idx[str(tw)]) = self.pick(tw)
            #
            meanUTC = UTCDateTime(np.array(list(_pt_float.values())).mean())
            medianUTC = UTCDateTime(np.median(
                            np.array(list(_pt_float.values()))))
            _pt_UTC['mean'] = meanUTC
            _pt_UTC['median'] = medianUTC
            # Convert floats into UTCDateTime
            for _kf, _vf in _pt_float.items():
                if _kf not in ('mean', 'median'):
                    _pt_UTC[_kf] = UTCDateTime(_vf)
        elif isinstance(self.time_win, (float, int)):
            _pt_float, _hos, _eval, _hos_idx = (
                                                    self.pick(self.time_win))
            _pt_UTC = UTCDateTime(_pt_float)
        else:
            logger.error("Param. time_win should be either iterable or float/int")
            raise HE.BadParameterValue()

        if debug_plot:
            HP.plot_HOST(self.tr,
                         _hos,
                         _eval,
                         _pt_UTC,
                         normalize=True,
                         plot_ax=None,
                         axtitle="HOST picks",
                         shift_cf=False,
                         plot_HOS=True,
                         plot_EVAL=True,
                         plot_intermediate_PICKS=True,
                         plot_final_PICKS=True,
                         show=True)

        # Store results in the class attribute
        self.pickTime_UTC = _pt_UTC
        self.hos_arr = _hos
        self.eval_fun = _eval
        self.hos_idx = _hos_idx

    def plot(self, **kwargs):
        """ Class wrapper for plotting the data of HOST) """
        outax = HP.plot_HOST(self.tr,
                             self.hos_arr,
                             self.eval_fun,
                             self.pickTime_UTC,
                             **kwargs,
                             show=True)
        return outax

    # ============================== Getter / Setter methods

    def get_picks_UTC(self):
        return self.pickTime_UTC

    def get_HOS(self):
        return self.hos_arr

    def get_eval_functions(self):
        return self.hos_arr

    def get_picks_index(self):
        return self.hos_idx

    def set_time_windows(self, timewin):
        # sliding windows checks
        if isinstance(timewin, (float, int)):
            self.time_win = (timewin,)
        elif isinstance(timewin, (list, tuple)):
            self.time_win = timewin
        else:
            logger.error("Input time windows is not a valid type")
            raise HE.BadInstance()

    def set_hos_method(self, method):
        if method.lower() in ('kurtosis', 'kurt', 'k'):
            self.method = "kurt"
        elif method.lower() in ('skewness', 'skew', 's'):
            self.method = "skew"
        else:
            logger.error("HOS method not valid ['kurtosis'/'skewness']")
            raise HE.BadParameterValue()

    def set_detection_method(self, method):
        if method.lower() in ('aic', 'akaike', 'a'):
            self.detection = "aic"
        elif method.lower() in ('diff', 'gauss'):
            self.detection = "diff"
        else:
            logger.error("DETECTION method Not valid ['aic'; 'diff'/'gauss']")
            raise HE.BadParameterValue()

    def set_diffgauss_threshold(self, threshold):
        if self.detection in ('diff', 'gauss'):
            self.thresh = threshold
        else:
            self.thresh = None
