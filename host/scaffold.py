import logging
import numpy as np
import pathlib
import ctypes as C
#
from host import errors as HE

logger = logging.getLogger(__name__)

""" HOST scaffold module

In this module are contained all the necessary functions
that let the picking algorithm work.

Most of them are needed to transform the CF for a better
recognition of arrival times (pick)

"""

# ===================================================== SETUP C LIBRARY
MODULEPATH = pathlib.Path(__file__).parent.absolute()
libname = tuple(MODULEPATH.glob("src/host_clib.*.so"))[0]
myclib = C.CDLL(libname)

# AIC
myclib.aicp.restype = C.c_int
myclib.aicp.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'), C.c_int,
                        # OUT
                        np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                        C.POINTER(C.c_int)]
# =====================================================


def _normalize_trace(work_list, rangeVal=[-1, 1]):
    """

    This simple method will normalize the trace between rangeVal.
    Simply by scaling everything...

    Args:
        work_list (list, tuple, numpy.ndarray): vector of values
        range_val (list, tuple): interval of transformation

    Returns:
        work_list (list, tuple, numpy.ndarray): input vector of values,
            trasnformed

    """

    minVal = min(work_list)
    maxVal = max(work_list)
    work_list[:] = [((x - minVal) / (maxVal - minVal)) *
                    (rangeVal[1] - rangeVal[0]) for x in work_list]
    work_list = work_list + rangeVal[0]
    return work_list


def _smooth(x, window_len=11, window='hanning'):
    """ Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the
    signal. The signal is prepared by introducing reflected copies of
    the signal (with the window size) in both ends so that transient
    parts are minimized in the begining and end part of the output
    signal.

    Args:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an
                    odd integer (samples)
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns:
        the smoothed signal

    Note:
        length(output) != length(input), to correct this:
        return y[(window_len/2-1):-(window_len/2)] instead of just y.

    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, ("Window is on of 'flat', 'hanning', 'hamming'," +
                            " 'bartlett', 'blackman'")
               )

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':    # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    #
    # 1) return y
    # 2) return y[(window_len/2-1):-(window_len/2)]
    return y[int(np.ceil(window_len/2-1)): -int(np.floor((window_len/2)))]


# ====================================  TRANSFORM CF

def transform_f2(inarr):
    """ Stair-like energy transformation

    Similar to transformation num 2 from Baillard et al. 2014

    Args:
        inarr (numpy.ndarray): input values 1D vector

    Returns:
        F2 (numpy.ndarray): modified input

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    F2 = np.zeros(inarr.size)
    F2[0] = inarr[0]
    for i in range(1, len(inarr)-1):
        if inarr[i+1]-inarr[i] >= 0:
            F2[i] = F2[i-1] + inarr[i+1] - inarr[i]
        else:
            F2[i] = F2[i-1]
    F2[-1] = F2[-2]
    return F2


def transform_f3(inarr):
    """ Removing linear trend

    Similar to transformation num 3 from Baillard et al. 2014

    Args:
        inarr (numpy.ndarray): input values 1D vector

    Returns:
        F3 (numpy.ndarray): modified input

    """

    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    from scipy.signal import detrend
    F3 = detrend(inarr)
    F3 = F3 - F3[0]  # bring first sample to 0
    return F3


def transform_f4(inarr):
    """ Pushing Down local minima

    Similar to transformation num 4 from Baillard et al. 2014

    Args:
        inarr (numpy.ndarray): input values 1D vector

    Returns:
        F4 (numpy.ndarray): modified input

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    Tk = [inarr[_xx] - np.max([inarr[_xx], inarr[_xx+1]])
          for _xx in range(0, (inarr.size-1))]
    F4 = [_ii if _ii <= 0.0 else 0.0 for _ii in Tk]
    F4 = np.append(F4, F4[-1])
    return np.array(F4)


def transform_f5(inarr, power=2.0):
    """ Elevate to power

    Elevate to the power of N (default 2) each single value in the array

    Args:
        inarr (numpy.ndarray): input values 1D vector

    Optional:
        power (float): input power

    Returns:
        F5 (numpy.ndarray): modified input

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    return inarr**power


def transform_smooth(inarr, smooth_win, window_type='hanning'):
    """ Simple smoothing of the CF to better extract the main pick.

    This function will smooth of a window equal to the one specified
    in sample

    Args:
        inarr (numpy.ndarray): 1D array with CFs
        smooth_win (int): sample size of smoothing window
        window_type (str): is the window used for the smoothing
            convolution. Possible choices: 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman' [default 'hanning']

    Note:
        smooth_win must be an ODD integer (after conversion). In case is
        EVEN, a +1 is given to transform it

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()

    if smooth_win % 2 == 0:
        smooth_win += 1
    return _smooth(inarr, window_len=smooth_win, window=window_type)


def gauss_dev(inarr, thr):
    """ Gaussian detector

    Assuming it's a gaussian process we calculate first-derivative of
    the input array. We declare the arrival at the first sample where
    its value exceed the input threshold (inarr[x] >= std*thr)

    Args:
        inarr (numpy.ndarray): input values 1D vector
        thr (float): detection threshold

    Returns:
        idx (int): index of pick detection
        m (float): mean
        s (float): standard deviation
        all_idx (numpy.ndarray): array of indexes where threshold is
            passed.
        hos_arr_diff (numpy.ndarray): discrete first derivative of the
            input array

    """
    hos_arr_diff = np.abs(np.diff(inarr))
    m = np.mean(hos_arr_diff)
    hos_arr_diff_zeromean = hos_arr_diff - m
    s = np.std(hos_arr_diff_zeromean)
    #
    all_idx = np.where(hos_arr_diff_zeromean >= thr*s)[0]
    try:
        return all_idx[0] - 1, m, s, all_idx, hos_arr_diff
    except IndexError:
        raise HE.PickNotFound("DIFF/GAUSS method failed to detect pick! " +
                              "Hint: try to lower the threshold")


def detect_minima(inarr):
    """ Simply return the INDEX POSITION of the minimum value and the
        input array as well.
    """
    return np.argmin(inarr), inarr


def AICcf(inarr):
    """ Call fast C-routin of AIC carachteristic funtion

    This method will return the index of the minimum AIC carachteristic
    function.

    Args:
        inarr (numpy.ndarray): input values 1D vector
        thr (float): detection threshold

    Returns:
        pminidx (int): index of absolute minima of AIC-CF
        aicfn (numpy.ndarray): AIC carachteristic function array

    """
    pminidx = C.c_int()
    tmparr = np.ascontiguousarray(inarr, np.float32)
    aicfn = np.zeros(inarr.size - 1,
                     dtype=np.float32, order="C")
    ret = myclib.aicp(tmparr, inarr.size,
                      aicfn, C.byref(pminidx))
    if ret != 0:
        raise MemoryError("Something wrong with AIC picker")
    #
    pminidx = pminidx.value
    return pminidx, aicfn
