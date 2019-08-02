import logging
import numpy as np
#
# from host import errors as HE

logger = logging.getLogger(__name__)

"""HOST scaffold module

In this module are contained all the necessary functions
that let the picking algorithm work.

Most of them are needed to calculate the CF and process it.

"""


def central_moment(arr, j, k=1, N=1):
    """
    Calculate the central moment of order k, given an array
     eq 16.17
    """
    if j <= N-1:
        sumup = j
    else:
        sumup = N
    return np.sum(arr[j-sumup+1:j+1] ** k)/N


def sixteen_eightteen(arr, j, k, N):
    """
    Calculate the central moment of order k, given an array
     eq 16.18
    """
    partA = central_moment(arr, j-1, k=k, N=N)
    if j < N:
        partB = (arr[0]**k + arr[j]**k)/N
    else:
        partB = (arr[j-N]**k + arr[j]**k)/N
    return (partA - partB)


def transform_f2(inarr):
    """
    Transformation num 2 from
    http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014
    """
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
    """
    Transformation num 3 from
    http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014
    """
    from scipy.signal import detrend
    return detrend(inarr)


def normalize_trace(workList, rangeVal=[-1, 1]):
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


def gauss_dev(inarr, thr):
    """
    Assuming it's half gaussian we calculate the mean and std.
    We multiply the std with a std deviation with a threshold
    """
    hos_arr_diff = np.diff(inarr)
    m = np.mean(hos_arr_diff)
    s = np.std(hos_arr_diff)
    #
    all_idx = np.where(hos_arr_diff >= thr*s)[0]
    return all_idx[0] - 1, m, s, all_idx, hos_arr_diff


def fit_and_pick(inarr, thr):
    """
    From this function, we expect to fit the exponential
    statistical distribution, and then operate with the mean/std
    threshold for detection of first arrival.
    """
    from scipy.optimize import curve_fit

    def fitFunc(t, a):
        return a*np.exp(-a*t)

    mydiff = np.diff(inarr)
    count, division = np.histogram(mydiff, bins=int(len(inarr)/2))
    fitParams, fitCov = curve_fit(fitFunc,
                                  division[0:len(division)-1],
                                  count)
    #
    mean = 1/fitParams[0]
    variance = 1/(fitParams[0]**2)
    all_idx = np.where(mydiff >= thr*mean)[0]
    return all_idx[0] - 1, mean, variance, fitCov, mydiff


def normalize_and_select(inarr, thr):
    """
    From this function, we expect to find the first arrival
    statistical distribution, and then operate with the mean/std
    threshold for detection of first arrival.
    """
    mydiff = np.diff(inarr)
    mydiff_norm = normalize_trace(mydiff, rangeVal=[0, 1])
    all_idx = np.where(mydiff_norm >= thr)[0]
    return all_idx[0] - 1, mydiff_norm, all_idx


def AICcf(td):
    """
    This method will return the index of the minimum AIC carachteristic
    function.

    td must be a  `numpy.ndarray`
    """
    # --------------------  Creation of the carachteristic function
    # AIC(k)=k*log(variance(x[1,k]))+(n-k+1)*log(variance(x[k+1,n]))
    AIC = np.array([])
    for ii in range(1, len(td)):
        with np.errstate(divide='raise'):
            try:
                var1 = np.log(np.var(td[0:ii]))
            except FloatingPointError:  # if var==0 --> log is -inf
                var1 = 0.00
            #
            try:
                var2 = np.log(np.var(td[ii:]))
            except FloatingPointError:  # if var==0 --> log is -inf
                var2 = 0.00
        #
        val1 = ii*var1
        val2 = (len(td)-ii-1)*var2    # ver2: +1 ver1(thison): -1
        AIC = np.append(AIC, (val1+val2))
    # -------------------- New idx search (avoid window's boarders)
    # (ascending order min->max) OK!
    idx = sorted(range(len(AIC)), key=lambda k: AIC[k])[0]

    # --- OLD (here for reference)
    # idxLst = sorted(range(len(AIC)), key=lambda k: AIC[k])
    # if idxLst[0]+1 not in (1, len(AIC)):  # need index. start from 1
    #     idx = idxLst[0]+1
    # else:
    #     idx = idxLst[1]+1

    # --- REALLY OLD idx search
    # idx_old=int(np.where(AIC==np.min(AIC))[0])+1
    # ****   +1 order to make multiplications
    # **** didn't take into account to minimum at the border of
    # **** the searching window
    return idx, AIC
