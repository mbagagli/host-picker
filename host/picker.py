"""HOST (Higher-Order Statistics) picking algorithms.

This module implements the main classes and functions used to detect
first arrivals using Higher-Order Statistics (HOS), specifically
kurtosis and skewness–based characteristic functions.

It wraps fast C routines for CF computation and provides a full
pipeline: minimal preprocessing, CF computation, optional CF
transformations, pick detection (AIC / Gaussian deviation / minima),
and uncertainty estimation (single-window SNR or multi-window robust
jackknife with MAD/Tukey outlier handling).

For end-to-end examples, see the Jupyter notebooks in the `books/`
directory.
"""

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
    """High-Order-Statistics picker.

    Encapsulates parameters and methods required to compute a HOS-based
    characteristic function (CF), apply optional CF transformations,
    detect onsets, and estimate uncertainty.

    Parameters
    ----------
    trace : obspy.core.Trace
        Input trace to analyze.
    time_windows : float | int | list[float] | tuple[float, ...]
        Analysis window length(s) in seconds for CF computation. A
        single value or a collection of values.
    hos_method : {"kurtosis","kurt","k","skewness","skew","s"}, optional
        CF flavor to compute. Default is "kurtosis".
    transform_cf : dict, optional
        Dictionary describing CF post-processing steps (see `scaffold`
        helpers). Keys are transformation names; values are parameter
        dicts. If empty, no transform is applied.
    detection_method : {"aic","akaike","a"} |
                       ("diff"|"gauss", threshold) |
                       {"min","minima"}, optional
        Pick declaration strategy. AIC (default), Gaussian deviation
        (requires a threshold), or local minima.

    Notes
    -----
    `time_windows` is normalized to a tuple internally. Private helpers
    are intended for internal usage; prefer public methods.
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
        """Minimal preprocessing of the active trace.

        Applies de-mean and linear de-trend to satisfy the zero-mean
        assumption behind HOS CFs and to stabilize downstream detection.
        """
        self.tr.detrend('demean')
        self.tr.detrend('simple')

    def _calculate_CF(self, tw):
        """Compute the HOS characteristic function (CF).

        Wraps optimized C routines to compute kurtosis/skewness CF over a
        sliding window.

        Parameters
        ----------
        tw : float
            Analysis window length in seconds.

        Returns
        -------
        hos_arr : numpy.ndarray
            CF array (float32), aligned to the input trace (initial `N`
            samples removed).
        N : int
            Window length in samples.

        Raises
        ------
        host.errors.MissingAttribute
            If the trace has no data.
        host.errors.BadParameterValue
            If `tw` is not greater than the trace sampling interval or if
            an invalid HOS method is requested.
        host.errors.PickNotFound
            If the C routine signals a computation failure.
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
        """Apply configured transformations to the CF.

        Chains transformations defined in `transform_cf` to enhance SNR
        before detection (e.g., smoothing, normalization).

        Parameters
        ----------
        inarr : numpy.ndarray
            Input CF array.
        num_sample : int
            Nominal window length in samples (useful to size transforms).

        Returns
        -------
        numpy.ndarray
            Transformed CF.

        Raises
        ------
        host.errors.BadInstance
            If `transform_cf` is not a dict.
        host.errors.MissingVariable
            If `inarr` is missing or empty.
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
        """Detect the pick from a CF using the selected strategy.

        Parameters
        ----------
        hos_arr : numpy.ndarray
            (Optionally transformed) CF array.

        Returns
        -------
        hos_idx : int
            Index of the declared pick within `hos_arr`.
        eval_fun : numpy.ndarray
            Evaluation function used by the detector (AIC curve, Gaussian
            deviation scores, or a proxy used by minima detection).

        Raises
        ------
        host.errors.MissingAttribute
            When the Gaussian deviation detector is selected without a
            threshold.
        host.errors.BadParameterValue
            On unsupported detection method.
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
        """Estimate pick quality via signal-to-noise ratio.

        Computes SNR as `signal / (2*std(noise))` to mitigate spikes:
        noise is the pre-pick window; signal is the mean of absolute max
        and min in the post-pick window.

        Parameters
        ----------
        pick_time : UTCDateTime
            Reference pick time.
        noise_win : float
            Noise window length in seconds (before `pick_time`).
        signal_win : float
            Signal window length in seconds (after `pick_time`).

        Returns
        -------
        float
            SNR value rounded to 2 decimals.
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

    def _jk(self, indict, seek_outliers=True, threshold_factor=1):
        """Estimate uncertainty via multi-window robust statistics.

        For ≥3 window estimates, identify outliers with a robust rule and
        compute the uncertainty as the span (max−min) across valid picks.
        If outlier search is disabled or dispersion is null, treat all
        observations as valid.

        Parameters
        ----------
        indict : dict[str, float]
            Map from window label (e.g., seconds as string) to pick time
            in POSIX seconds (float).
        seek_outliers : bool, optional
            If False, skip outlier detection and treat all as valid.
        threshold_factor : float, optional
            Robust threshold multiplier. Interpreted as:
            - MAD rule: accept |x−median| / (1.4826*MAD) ≤ `threshold_factor`
            - Tukey fallback: accept within Q1±`threshold_factor`*IQR

        Returns
        -------
        pick_error : float
            Uncertainty window in seconds (rounded to 2 decimals).
        valid_obs_dict : dict[str, UTCDateTime]
            Sorted valid observations converted to UTCDateTime.
        outliers_dict : dict[str, UTCDateTime]
            Sorted outlier observations converted to UTCDateTime.

        Notes
        -----
        When MAD collapses to zero (ties), Tukey's fences are used as a
        fallback. If both dispersion measures collapse, all observations
        are treated as valid.
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

            if only_zero or not seek_outliers:
                # No bias, everybody on the same pick
                logger.debug("All picks agree ... or no-seek for outliers!")
                _lv = tuple(indict.values())
                pick_error = np.max(_lv) - np.min(_lv)
                pick_error += 0.00111
                pick_error = np.round(pick_error, 2)
                #
                valid_obs_dict = {k: UTCDateTime(v) for k, v in indict.items()}
                outliers_dict = {}

            else:
                # There's changes, check outliers / valid (ROBUST: MAD + Tukey fallback)
                logger.debug("There's pick variation -> checking for outlier (MAD + Tukey fallback)")

                # Collect keys and values
                keys = list(indict.keys())
                vals = np.array([float(indict[k]) for k in keys], dtype=float)

                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                robust_sigma = 1.4826 * mad

                if robust_sigma == 0.0:
                    # Tukey fallback
                    q1, q3 = np.percentile(vals, [25, 75])
                    iqr = q3 - q1
                    if iqr == 0.0:
                        valid_obs = list(zip(keys, vals))
                        outliers = []
                    else:
                        lo = q1 - threshold_factor * iqr
                        hi = q3 + threshold_factor * iqr
                        valid_mask = (vals >= lo) & (vals <= hi)
                        valid_obs = [(k, v) for k, v, m in zip(keys, vals, valid_mask) if m]
                        outliers = [(k, v) for k, v, m in zip(keys, vals, valid_mask) if not m]
                else:
                    z = np.abs(vals - med) / robust_sigma
                    valid_mask = z <= threshold_factor
                    valid_obs = [(k, v) for k, v, m in zip(keys, vals, valid_mask) if m]
                    outliers = [(k, v) for k, v, m in zip(keys, vals, valid_mask) if not m]

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
                    outliers_dict = {k: UTCDateTime(v) for k, v in indict.items()}
                else:
                    # -- Calc Error
                    _only_valid = [_t[1] for _t in valid_obs]
                    pick_error = np.max(_only_valid) - np.min(_only_valid)

                    # -- Round Error
                    pick_error += 0.000111
                    pick_error = np.round(pick_error, 3)

                    valid_obs_dict = dict(sorted(valid_obs, key=lambda x: x[1]))
                    valid_obs_dict = {k: UTCDateTime(v) for k, v in valid_obs_dict.items()}

                    outliers_dict = dict(sorted(outliers, key=lambda x: x[1]))
                    outliers_dict = {k: UTCDateTime(v) for k, v in outliers_dict.items()}

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
        """Return mean and median pick times from a dict of UTC floats.

        Parameters
        ----------
        indict : dict[str, float]
            Mapping of labels to POSIX-second pick times.

        Returns
        -------
        meanUTC : UTCDateTime | None
            Mean pick time, or None if `indict` is empty.
        medianUTC : UTCDateTime | None
            Median pick time, or None if `indict` is empty.
        """
        if len(indict) == 0:
            return None, None
        else:
            datearr = [float(_v) for _k, _v in indict.items()]
            meanUTC = UTCDateTime(np.array(datearr).mean())
            medianUTC = UTCDateTime(np.median(np.array(datearr)))
            return meanUTC, medianUTC

    def _pick(self, tw):
        """Compute CF, detect pick, and return core artifacts for a window.

        Parameters
        ----------
        tw : float
            Analysis window length in seconds.

        Returns
        -------
        pickTime_UTC : float
            Pick time as POSIX seconds (float).
        hos_arr : numpy.ndarray
            CF array (possibly transformed).
        eval_fun : numpy.ndarray
            Evaluation function used by the detector.
        hos_idx : int
            Pick index (offset includes the initial window shift `N`).

        Notes
        -----
        The float timestamp is used in downstream statistics to avoid
        repeated datetime conversions.
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

    def work(self, debug_plot=False, noise_win=1.0, signal_win=1.0,
             seek_outliers=True, threshold_factor=1):
        """Run the full HOST pipeline and store results on the instance.

        Steps:
          1) For each `time_windows` entry, compute CF and detect a pick.
          2) Convert float pick times to UTCDateTime.
          3) If multiple windows: estimate uncertainty via `_jk`
             (robust outlier handling). If single window: estimate SNR
             via `_snratio`.
          4) Aggregate mean and median from valid observations.

        Parameters
        ----------
        debug_plot : bool, optional
            If True, render a debug plot of CFs and picks.
        noise_win : float, optional
            Noise window length (seconds) for SNR when using a single
            window.
        signal_win : float, optional
            Signal window length (seconds) for SNR when using a single
            window.
        seek_outliers : bool, optional
            Enable robust outlier removal across windows.
        threshold_factor : float, optional
            Robust threshold multiplier (see `_jk`).

        Notes
        -----
        Results are written to:
            - `self.pickTime_UTC` (dict with picks, error, valid/outliers)
            - `self.hos_arr` (CFs per window)
            - `self.eval_fun` (evaluation functions)
            - `self.hos_idx` (indices)
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
                 _pt_UTC['outlier_obs']) = self._jk(
                                            _pt_float,
                                            seek_outliers=seek_outliers,
                                            threshold_factor=threshold_factor)
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
        """Plot convenience wrapper.

        Delegates to `host.plotting.plot_HOST` and returns the axes. All
        keyword arguments are forwarded to the plotting helper.
        """
        outax = HPL.plot_HOST(self, **kwargs, show=True)
        return outax

    # ------------- Getter / Setter methods

    def calculate_single_hos(self, tw, shift_origin=True):
        """Compute a single-window CF for inspection or debugging.

        Parameters
        ----------
        tw : float
            Analysis window length in seconds.
        shift_origin : bool, optional
            If False, pre-pend the first CF value `N` times so the CF has
            the same length as the input trace. Not recommended for picking
            as it can bias onset detection.

        Returns
        -------
        numpy.ndarray
            CF array (possibly origin-aligned based on `shift_origin`).
        """
        hos_arr, N = self._calculate_CF(tw)
        if not shift_origin:
            _add_me = np.full(N, hos_arr[0])
            hos_arr = np.insert(hos_arr, 0, _add_me, axis=0)
        return hos_arr

    def get_picks_UTC(self):
        """Return the dictionary of picks and summary stats."""
        return self.pickTime_UTC

    def get_HOS(self):
        """Return the computed CF arrays per time window."""
        return self.hos_arr

    def get_eval_functions(self):
        """Return evaluation functions used during detection."""
        return self.eval_fun

    def get_picks_index(self):
        """Return pick indices per window (post-shift)."""
        return self.hos_idx

    def set_time_windows(self, timewin):
        """Set analysis window(s).

        Parameters
        ----------
        timewin : float | int | list[float] | tuple[float, ...]
            New window length(s) in seconds.
        """
        if isinstance(timewin, (float, int)):
            self.time_win = (timewin,)
        elif isinstance(timewin, (list, tuple)):
            self.time_win = timewin
        else:
            logger.error("Input time windows is not a valid type")
            raise HE.BadInstance()

    def set_hos_method(self, method):
        """Set the HOS CF method.

        Parameters
        ----------
        method : {"kurtosis","kurt","k","skewness","skew","s"}
            Desired CF flavor.
        """
        if method.lower() in ('kurtosis', 'kurt', 'k'):
            self.method = "kurt"
        elif method.lower() in ('skewness', 'skew', 's'):
            self.method = "skew"
        else:
            logger.error("HOS method not valid ['kurtosis'/'skewness']")
            raise HE.BadParameterValue()

    def set_detection_method(self, method):
        """Set the pick detection strategy.

        Parameters
        ----------
        method : {"aic","akaike","a","diff","gauss","min","minima"}
            Detection method. For "diff"/"gauss", set the threshold via
            `set_diffgauss_threshold`.
        """
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
        """Set or replace the CF-transformation pipeline.

        Parameters
        ----------
        transform_dict : dict
            Mapping from transformation name to parameter dict used by
            `scaffold` helpers.

        Raises
        ------
        host.errors.BadParameterValue
            If `transform_dict` is not a dict.
        """
        if isinstance(transform_dict, dict):
            self.tr_cf = transform_dict
        else:
            logger.error("transform_dict must a a dictionary!")
            raise HE.BadParameterValue()

    def set_diffgauss_threshold(self, threshold):
        """Set the threshold for the Gaussian deviation detector.

        Parameters
        ----------
        threshold : float
            Threshold value used by the "diff"/"gauss" detection mode.
        """
        if self.detection in ('diff', 'gauss'):
            self.thresh = threshold
        else:
            self.thresh = None
