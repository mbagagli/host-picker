import logging
import numpy as np
import pathlib
import ctypes as C
#
from host import errors as HE

logger = logging.getLogger(__name__)

"""HOST scaffold module

In this module are contained all the necessary functions
that let the picking algorithm work.

Most of them are needed to calculate the CF and process it.

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


def _normalize_trace(workList, rangeVal=[-1, 1]):
    ''' This simple method will normalize the trace between rangeVal.
        Simply by scaling everything...
        *** INPUT MUST BE A list/tuple object

    '''
    minVal = min(workList)
    maxVal = max(workList)
    workList[:] = [((x - minVal) / (maxVal - minVal)) *
                   (rangeVal[1] - rangeVal[0]) for x in workList]
    workList = workList + rangeVal[0]
    return workList


def _smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the
    signal. The signal is prepared by introducing reflected copies of
    the signal (with the window size) in both ends so that transient
    parts are minimized in the begining and end part of the output
    signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an
                    odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array
          instead of a string.

    NOTE: length(output) != length(input), to correct this:
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
    """It's equal to transformation num 2 from
    http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    b = np.zeros(inarr.size)
    b[0] = inarr[0]
    for i in range(1, len(inarr)-1):
        if inarr[i+1]-inarr[i] >= 0:
            b[i] = b[i-1] + inarr[i+1] - inarr[i]
        else:
            b[i] = b[i-1]
    b[-1] = b[-2]
    return b


def transform_f3(inarr):
    """Removing linear trend

    It's equal to transformation num 3 from
    http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014

    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    from scipy.signal import detrend
    outa = detrend(inarr)
    outa = outa - outa[0]  # bring first sample to 0
    return outa


def transform_f4(inarr):
    """Pushing Down local minima

    It's equal to transformation num 4 from
    http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014

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
    """Elevate each single value in the array to the specified power
    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    return inarr**power


def transform_smooth(inarr, smooth_win, window_type='hanning'):
    """Simple smoothing of the CF to better extract  the main transient.

    smooth_win is in SAMPLE!
    If smooth_win is even a +1 is given to make it odd
    """
    if not isinstance(inarr, np.ndarray):
        logger.error("The input vector should be a numpy array type")
        raise HE.BadInstance()
    #
    if smooth_win % 2 == 0:
        smooth_win += 1
    return _smooth(inarr, window_len=smooth_win, window=window_type)


def gauss_dev(inarr, thr):
    """
    Assuming it's half gaussian we calculate the mean and std.
    We multiply the std with a std deviation with a threshold
    """
    # hos_arr_diff = np.diff(inarr)
    hos_arr_diff = np.abs(np.diff(inarr))
    m = np.mean(hos_arr_diff)
    s = np.std(hos_arr_diff)
    #
    all_idx = np.where(hos_arr_diff >= thr*s)[0]
    all_idx[0]
    try:
        return all_idx[0] - 1, m, s, all_idx, hos_arr_diff
    except IndexError:
        raise HE.PickNotFound()


def detect_minima(inarr):
    """ Simply return the INDEX POSITION of the minimum value
        for the given array
    """
    return np.argmin(inarr), inarr


def AICcf(td):
    """
    This method will return the index of the minimum AIC carachteristic
    function.

    td must be a  `numpy.ndarray`
    """
    pminidx = C.c_int()
    tmparr = np.ascontiguousarray(td, np.float32)
    aicfn = np.zeros(td.size - 1,
                     dtype=np.float32, order="C")
    ret = myclib.aicp(tmparr, td.size,
                      aicfn, C.byref(pminidx))
    if ret != 0:
        raise MemoryError("Something wrong with AIC picker")
    #
    pminidx = pminidx.value
    return pminidx, aicfn
