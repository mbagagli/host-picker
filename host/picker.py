import logging
import numpy as np
import pathlib
import ctypes as C
#
from obspy.core import Trace
from obspy.core import UTCDateTime
#
from host import errors as HE
from host import plotting as HPL
from host import scaffold as HS

logger = logging.getLogger(__name__)


""" Main HOST picking algorithm module

Here are stored all the functions and classes needed to pick
with Higher-Order-Statistics algorithms (skewness/kurtosis).

For a detailed explanation of the usage and example, the user is
referred to the Jupyter notebooks contained in `books` subdir.

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

# ===================================================== Additional ...
DEFAULTPICKERROR = -9999.99


class Host(object):
    """ Main class for the High Order STatistics picker

    This class contains all the necessary parameter to run the picker
    and defining the obtained pick

    Args:
        stream (obspy.core.Stream): obspy stream object containing the
            interested traces
        time_windows (int, float, list, tuple): can be either a single
            time window (in seconds) or a series of them. This time
            window represent the window of analysis for the HOS-CF
            calculations.
        hos_method (str): The basic characteristic function to use. It
            be either `'kurtosis' ['kurt', 'k']` (default) or
            `'skewness' ['skew', 's']`
        transform_cf (dict): Dictionary containing all the necessary
            transformation functions for the CF. For a full list and
            exaples, please check the jupyter notebook tutorials under
            the `book` subdir.
        detection_method (str): pick declaration method. Either
            `'aic' ['akaike', 'a']` (default), `'diff' ['gauss']`, or
            `'min' ['minima']`. For example of usage, please check the
            jupyter notebook tutorials under the `book` subdir.

    Note:
        The `time_window` instance will be transformed internally into
        class attribute of type `list` or `tuple`.

    Warning:
        The private methods are ment for developers only! Please
        refer to the public ones instead. Unexpected behavior may occur.

    """
    def __init__(self,
                 trace,
                 time_windows,
                 hos_method="kurtosis",
                 transform_cf={},
                 detection_method="aic"):

        # stream checks
        if isinstance(trace, Trace):
            self.tr = trace.copy()
            self.dt = self.tr.stats.delta
            self.ts = self.tr.data
        else:
            logger.error("Not a valid ObsPy Trace instance")
            raise HE.BadInstance()

        # sliding windows checks -->
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
        if (isinstance(detection_method, str) and
           detection_method.lower() in ('aic', 'akaike', 'a')):
            self.detection = "aic"
            self.thresh = None
        elif (isinstance(detection_method, (list, tuple)) and
              detection_method[0].lower() in ('diff', 'gauss')):
            self.detection = "gauss"
            self.thresh = detection_method[1]
        elif (isinstance(detection_method, str) and
              detection_method.lower() in ('min', 'minima')):
            self.detection = "minima"
            self.thresh = None
        else:
            logger.error("DETECTION method Not valid !"
                         "['aic', 'gauss', 'minima']")
            raise HE.BadParameterValue()

        if transform_cf:
            self.tr_cf = transform_cf
        else:
            self.tr_cf = None

        # Initialize output attributes
        self.pickTime_UTC = {}
        self.pickTime_error = {}
        self.hos_arr = {}
        self.eval_fun = {}
        self.hos_idx = {}
        self.work_picks = {}

    # ====================================== PRIVATE methods
    # ======================================================

    def _preprocess(self):
        """ Private method: minimal preprocessing

        This class-method performs a de-mean and a linear de-trend to
        the active/seleceted trace in the stream. This is needed for the
        0-mean assumption of HOS based picker and therefore their CF
        calculation.

        """
        self.tr.detrend('demean')
        self.tr.detrend('simple')

    def _calculate_CF(self, tw):
        """ Private method: calculate the carachteristic function

        This method-class is a wrapper for calling the C-routines
        for either kurtosis and skewness.

        Args:
            tw (float): time window in seconds for HOS-CF calculations

        Returns:
            hos_arr (numpy.ndarray): trace HOS-CF
            N (int): length in samples of the time-window of analysis

        """
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
        """ Modify the CF array.

        This private method will take care of the transformation
        of the HOST CF function to better increase the signal/noise
        ratio and therefore leading to a better pick detection.

        Args:
            inarr (numpy.ndarray): the HOS-CF array
            num_sample (int): number of samples for smoothing window

        """
        if isinstance(inarr, np.ndarray) and inarr.size != 0:
            if self.tr_cf and isinstance(self.tr_cf, dict):
                outarr = inarr
                for _kk, _vv in self.tr_cf.items():

                    logger.debug("Transform HOST CF: %s" % _kk)

                    if _kk.lower() == 'transform_smooth':

                        call_funct = getattr(HS, 'transform_smooth')
                        outarr = call_funct(outarr, num_sample, **_vv)

                    elif _kk.lower() == 'transform_smooth_custom':
                        # Check if win-size is higher than CF length
                        win_sample = np.int(
                            _vv['smooth_win'] / self.tr.stats.delta)

                        if win_sample >= outarr.size:
                            logger.warning(("The %d (samples) window " +
                                            "specified is longer than CF: " +
                                            "%d (samples) !!! " +
                                            "SKIPPING !!!") % (
                                              win_sample, outarr.size))
                        else:
                            call_funct = getattr(HS, 'transform_smooth')
                            outarr = call_funct(outarr, win_sample,
                                                **{x: y for x, y in _vv.items()
                                                   if x != 'smooth_win'})

                    else:
                        # Standard call
                        call_funct = getattr(HS, _kk.lower())
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
        """ Use the evalutation functions to declare picks

        Use one of the possible method to detect the pick over an
        HOS carachteristic function.

        Args:
            hos_arr (numpy.ndarray): the HOS-CF array (already
                transformed)

        """
        logger.debug("Detecting PICK with %s" % self.detection.upper())
        if self.detection.lower() in ('diff', 'gauss'):
            if not self.thresh:
                logger.error("Missing threshold for 'diff' DETECTION method")
                raise HE.MissingAttribute()
            hos_idx, m, s, all_idx, eval_fun = HS.gauss_dev(hos_arr,
                                                            self.thresh)

        elif self.detection.lower() in ('aic', 'akaike'):
            hos_idx, eval_fun = HS.AICcf(hos_arr)

        elif self.detection.lower() in ('min', 'minima'):
            hos_idx, eval_fun = HS.detect_minima(hos_arr)

        else:
            logger.error("!WEIRD! Invalid pick extr. mode: %s " +
                         "['aic', 'gauss', 'minima']" % self.detection)
            raise HE.BadParameterValue()
        #
        return hos_idx, eval_fun

    def _snratio(self, pick_time, noise_win, signal_win):
        """ Using signal-to-noise ratio to define pick error

        This method will calculate and report the signal-to-noise ratio
        as an indicator of the pick quality.
        In order to mitigate spiky traces, the noise value will be
        set as 2*std(noise). Also to mitigate spikes, the signal value
        will be estimated as the mean value among the absolute maximum
        and minimum of the data contained into the window.

        Note:
            This method is called only when a single pick (or evaluation
            window) is given. If multi-window approach is adopted, the
            pick-error will be evaluated via the jack-knife statistical
            approach.

        Args:
            pick_time (UTCDateTime): reference time for time-windows
                selection
            noise_win (int, float): seconds of noise window duration
            signal_win (int, float): seconds of signal window duration

        Returns:
            s2nr (float): the signal to noise ratio

        """
        nw = self.tr.slice(pick_time-noise_win, pick_time)
        sw = self.tr.slice(pick_time, pick_time+signal_win)
        #
        nv = np.std(nw.data)*2
        sv = (np.max(sw.data) + np.abs(np.min(sw.data))) / 2
        #
        s2nr = sv/nv

        # -- Round Error
        # trick/workaround to properly round  XXX.5 to XXX+1.0  (MB)
        # numpy e python round to the NEAREST EVEN number (i.e. 2.5 --> 2)
        s2nr += 0.00111
        s2nr = np.round(s2nr, 2)

        return s2nr

    def _jk(self, indict):
        """ Using jack-knife to define outliers and error

        This method discriminate between valid and outlier observations,
        and asses the pick-error as the time span among the valid-picks.

        Args:
            indict (dict): dictionary containing the datetime seconds
                of each pick (floats). The respective key represents
                the evaluation window length (in seconds)

        Returns:
            pick_error (float): uncertainty window in seconds
            valid_obs_dict (dict): contains the **sorted** valid
                observations as UTCDateTime object
            outliers_dict (dict): contains the **sorted** outlier
                observations as UTCDateTime object

        """
        valid_obs = []    # will be transformed in dict for output
        outliers = []     # will be transformed in dict for output

        if len(indict) >= 3:
            replicates = {}
            orig_mean = np.mean(tuple(indict.values()))

            # kk is the LEFT OUT obs
            for kk, vv in indict.items():
                ta = [indict[i] for i in indict.keys() if i != kk]
                replicates[kk] = {'mean': np.mean(ta),
                                  'bias': np.mean(ta) - orig_mean
                                  }

            # Standard deviation of residuals (BIAS)
            jkn_bias_std = np.std([vv['bias'] for kk, vv in
                                   replicates.items()])
            only_zero = np.all((jkn_bias_std == 0.0))

            if only_zero:
                # No bias, everybody on the same pick
                logger.debug("All picks agree ...")
                _lv = tuple(indict.values())
                pick_error = np.max(_lv) - np.min(_lv)
                pick_error += 0.00111
                pick_error = np.round(pick_error, 2)
                #
                valid_obs_dict = {k: UTCDateTime(v) for k, v in indict.items()}
                outliers_dict = {}

            else:
                # There's changes, check outliers / valid
                logger.debug("There's pick variation -> checking for outlier")

                # Evaluating the 2*std threshold
                for kk, vv in replicates.items():
                    if not np.abs(vv['bias']) >= jkn_bias_std:
                        # It's a valid observation
                        valid_obs.append((kk, indict[kk]))
                    else:
                        # It's a complete outlier
                        outliers.append((kk, indict[kk]))

                if not valid_obs:
                    """  No valid obs, because extreme change in BIAS.
                    e.g. Bulk obs balanced at the edges.
                         Although should never enter here, because even
                         if 2 obs and 2 obs at the edge, create a wide
                         gaussian distribution that.will invude them
                         anyway inside the 2*std threshold
                    This switch is just here for stability reason ...
                    """
                    logger.warning("No valid-obs found! Picks unstable! " +
                                   "Setting PICK-ERROR to DEFAULT [%.2f]" %
                                   DEFAULTPICKERROR)
                    pick_error = DEFAULTPICKERROR
                    valid_obs_dict = {}
                    outliers_dict = {
                        k: UTCDateTime(v) for k, v in indict.items()}

                else:
                    # -- Calc Error
                    _only_valid = [_t[1] for _t in valid_obs]
                    pick_error = np.max(_only_valid) - np.min(_only_valid)

                    # -- Round Error
                    pick_error += 0.00111
                    pick_error = np.round(pick_error, 2)

                    # Now valid Obs and Outlier could be DICT
                    valid_obs_dict = dict(
                        sorted(valid_obs, key=lambda x: x[1]))
                    valid_obs_dict = {k: UTCDateTime(v) for k, v in
                                      valid_obs_dict.items()}

                    outliers_dict = dict(sorted(outliers, key=lambda x: x[1]))
                    outliers_dict = {k: UTCDateTime(v) for k, v in
                                     outliers_dict.items()}
        else:
            _lv = tuple(indict.values())
            pick_error = np.max(_lv) - np.min(_lv)
            pick_error += 0.00111
            pick_error = np.round(pick_error, 2)
            #
            valid_obs_dict = {k: UTCDateTime(v) for k, v in indict.items()}
            outliers_dict = {}

        return pick_error, valid_obs_dict, outliers_dict

    def _define_final_pick(self, indict):
        """ Simply return mean and median UTCDateTime of indict
            UTCDateTime obkject. It ignores any keys
        """
        datearr = [float(_v) for _k, _v in indict.items()]
        meanUTC = UTCDateTime(np.array(datearr).mean())
        medianUTC = UTCDateTime(np.median(np.array(datearr)))
        return meanUTC, medianUTC

    def _pick(self, tw):
        """ Private method extracting picks

        This class method is called by the `work` method. It first
        calculate the CF, and eventually detect the first arrival.

        Args:
            tw (float): time window in seconds for HOS-CF calculations

        Returns:
            pickTime_UTC (float): UTC time-float of the pick. Having
                a float in return is necessary for further statistical
                calculation. The conversion will occur inside the
                calling method
            hos_arr (numpy.ndarray): array of the calculated CF
            eval_fun (numpy.ndarray): array of the evaluation function
                adopted to declare the pick
            hos_idx (int): index of the CF declared pick

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
        # Returning float is necessary to calculate MEAN, MEDIAN and ERR
        return float(pickTime_UTC), hos_arr, eval_fun, (hos_idx+N)

    # ====================================== PUBLIC methods
    # ======================================================

    def work(self, debug_plot=False, noise_win=1.0, signal_win=1.0):
        """Main public method to run the picking algotrithm

        This represent the core method for the Host class. In order,
        it will calculate

        Optional:
            noise_win (int, float): noise window length (in seconds)
                used to assess the signal to noise ration/
            debug_plot (bool): if True, the method will return a
                floating, interactive figure with the picking results

        Note:
            The noise_win and signal_win options will be used by the
            `_snratio` method only when a single-window is used.

        """
        # Check time windows
        if isinstance(self.time_win, (list, tuple)):
            _pt_UTC = {}
            _pt_float = {}
            _hos = {}
            _eval = {}
            _hos_idx = {}
            # === 1. Pick
            for tw in self.time_win:
                (_pt_float[str(tw)],
                 _hos[str(tw)],
                 _eval[str(tw)],
                 _hos_idx[str(tw)]) = self._pick(tw)

            # === 2. Convert results-floats into UTCDateTime
            _pt_UTC = {_kf: UTCDateTime(_vf)
                       for (_kf, _vf) in _pt_float.items()}

            # === 3. Declare the error
            if len(self.time_win) >= 2:
                (_pt_UTC['pick_error'],
                 _pt_UTC['valid_obs'],
                 _pt_UTC['outlier_obs']) = self._jk(_pt_float)
            else:
                _pt_UTC['pick_error'] = self._snratio(
                    _pt_UTC[str(self.time_win[0])], noise_win, signal_win)
                _pt_UTC['valid_obs'] = {
                    str(self.time_win[0]): _pt_UTC[str(self.time_win[0])]}
                _pt_UTC['outlier_obs'] = {}

            # === 4. Declare pick
            _pt_UTC['mean'], _pt_UTC['median'] = self._define_final_pick(
                    _pt_UTC['valid_obs'])

        else:
            logger.error("Parameter time_windows should be either " +
                         "iterable or float/int")
            raise HE.BadParameterValue()

        # Store results in the class attribute
        self.pickTime_UTC = _pt_UTC
        self.hos_arr = _hos
        self.eval_fun = _eval
        self.hos_idx = _hos_idx

        if debug_plot:
            _ = HPL.plot_HOST(self,
                              normalize=True,
                              debug_plot=True,
                              plot_final_picks=True,
                              axtitle="HOST picks",
                              show=True)

    def plot(self, **kwargs):
        """ Class wrapper for plotting the data of HOST """
        outax = HPL.plot_HOST(self, **kwargs, show=True)
        return outax

    # ------------- Getter / Setter methods

    def get_picks_UTC(self):
        return self.pickTime_UTC

    def get_HOS(self):
        return self.hos_arr

    def get_eval_functions(self):
        return self.eval_fun

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
        elif method.lower() in ('min', 'minima'):
            self.detection = "min"
        else:
            logger.error("DETECTION method Not valid ['aic'; 'diff'/'gauss']")
            raise HE.BadParameterValue()

    def set_transform_cf(self, transform_dict):
        """ Modify or set the CF-transformation dict

        Args:
            transform_dict (dict): key/value dictionary containing as
                key the name of one function transformation contained
                in scaffold module, as value another dictionary for
                the functions parameters

        """
        if isinstance(transform_dict, dict):
            self.tr_cf = transform_dict
        else:
            logger.error("transform_dict must a a dictionary!")
            raise HE.BadParameterValue()

    def set_diffgauss_threshold(self, threshold):
        if self.detection in ('diff', 'gauss'):
            self.thresh = threshold
        else:
            self.thresh = None
