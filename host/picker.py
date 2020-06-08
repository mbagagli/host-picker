import logging
import numpy as np
import pathlib
import ctypes as C
#
from obspy.core import Stream
from obspy.core import UTCDateTime
#
from host import errors as HE
from host import plotting as HPL
from host import scaffold as HS

logger = logging.getLogger(__name__)


"""Main picker module of HOST picking algorithm

In this module are stored all the functions and classes needed to pick
with HOS algorithms (skewness/kurtosis).

For a detailed explanation of the usage, the user should look the docs.

"""

# ===================================================== SETUP C LIBRARY
MODULEPATH = pathlib.Path(__file__).parent.absolute()
libname = tuple(MODULEPATH.glob("src/host_clib.*.so"))[0]
myclib = C.CDLL(libname)

myclib.kurtcf.restype = C.c_int
myclib.kurtcf.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                          C.c_int, C.c_int,
                          # OUT
                          np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]
#
myclib.kurtcf_mean.restype = C.c_int
myclib.kurtcf_mean.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                               C.c_int, C.c_int,
                               # OUT
                               np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]
#
myclib.skewcf.restype = C.c_int
myclib.skewcf.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                          C.c_int, C.c_int,
                          # OUT
                          np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]
#
myclib.skewcf_mean.restype = C.c_int
myclib.skewcf_mean.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                               C.c_int, C.c_int,
                               # OUT
                               np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]

# =====================================================================


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
                 transform_cf={},
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

        if transform_cf:
            self.tr_cf = transform_cf
        else:
            self.tr_cf = None

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
        logger.debug("Calculating CF with HOS: %s" % self.method.upper())
        N = round(tw/self.dt) + 1
        tmparr = np.ascontiguousarray(self.ts, np.float32)
        hos_arr = np.zeros(self.ts.size, dtype=np.float32, order="C")
        if self.method == "kurt":
            ret = myclib.kurtcf(tmparr, self.ts.size, N, hos_arr)
        elif self.method == "kurt_mean":
            ret = myclib.kurtcf_mean(tmparr, self.ts.size, N, hos_arr)
        elif self.method == "skew":
            ret = myclib.skewcf(tmparr, self.ts.size, N, hos_arr)
        elif self.method == "skew_mean":
            ret = myclib.skewcf_mean(tmparr, self.ts.size, N, hos_arr)
        else:
            logger.error("!WEIRD! Wrong HOS method given " +
                         "['skew', 'skew_mean', 'kurt', 'kurt_mean']")
            raise HE.BadParameterValue()
        #
        if ret != 0:
            raise HE.PickNotFound("CF calculation went wrong! [%s]",
                                  self.method)
        hos_arr = np.delete(hos_arr, np.arange(N))
        return hos_arr, N

    def _transform_cf(self, inarr, num_sample):
        """ This private method will take care of the transformation
            of the HOST CF function to better increase the signal/noise
            ratio and therefore leading to a better pick detection.

        """
        if isinstance(inarr, np.ndarray) and inarr.size != 0:
            if self.tr_cf and isinstance(self.tr_cf, dict):
                outarr = inarr
                for _kk, _vv in self.tr_cf.items():
                    logger.debug("Transform HOST CF: %s" % _kk)
                    call_funct = getattr(HS, _kk.lower())
                    if _kk.lower() == 'transform_f4':
                        outarr = call_funct(outarr, num_sample, **_vv)
                    else:
                        outarr = call_funct(outarr, **_vv)
            else:
                logger.error("The transform_cf parameter must be a dict!")
                raise HE.BadInstance()
        else:
            logger.error("Missing INPUT array")
            raise HE.MissingVariable()
        #
        return outarr

    def _detect_pick(self, hos_arr):
        """Use one of the possible method to detect the pick over an
           HOS carachteristic function.

        """
        logger.debug("Detecting PICK with %s" % self.detection.upper())
        if self.detection.lower() in ('diff', 'gauss'):
            if not self.thresh:
                logger.error("Missing threshold for 'diff' DETECTION method")
                raise HE.MissingAttribute()
            try:
                hos_idx, m, s, all_idx, eval_fun = HS.gauss_dev(hos_arr,
                                                                self.thresh)
            except HE.PickNotFound:
                logger.error("Critical error! We should not be here! " +
                             " The hos_arr should be always positive, " +
                             "and therefore also the threshold level")
                HS._abort()

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

        # ======================== Calculate CF
        hos_arr, N = self._calculate_CF(tw)

        # ======================== Transform CF
        if self.tr_cf:
            hos_arr = self._transform_cf(hos_arr, N)

        # ======================== Extract Pick (AIC/GAUSS)
        hos_idx, eval_fun = self._detect_pick(hos_arr)

        # ======================== Closing
        # time= NUMsamples/df OR NUMsamples*dt
        logger.debug("HOS: %s - PICKSEL: %s - idx: %r" % (self.method,
                                                          self.detection,
                                                          hos_idx+N))
        pickTime_UTC = self.tr.stats.starttime + ((hos_idx+N) * self.dt)
        #
        return float(pickTime_UTC), hos_arr, eval_fun, (hos_idx+N)

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
            _pt_float, _hos, _eval, _hos_idx = self.pick(self.time_win)
            _pt_UTC = UTCDateTime(_pt_float)

        else:
            logger.error("Parameter time_windows should be either " +
                         "iterable or float/int")
            raise HE.BadParameterValue()

        if debug_plot:
            HPL.plot_HOST(self.tr,
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
        """ Class wrapper for plotting the data of HOST """
        outax = HPL.plot_HOST(self.tr,
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

