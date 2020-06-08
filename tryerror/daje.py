#!/usr/bin/env python

# --- Compiler call
# gcc -shared -rdynamic aurem_c.c -o aurem_c.so

from obspy import read
import numpy as np
import ctypes as C
import pathlib
import matplotlib.pyplot as plt


#libname= "/home/matteo/miniconda3/envs/aurem/lib/python3.6/site-packages/aurem_clib.cpython-36m-x86_64-linux-gnu.so"
libname = pathlib.Path().absolute()/"host_clib.so"
myclib = C.CDLL(libname)

print(libname)


st = read()
tr = st[0]
tr.detrend("linear")
tr.detrend("demean")
tr.filter("highpass", freq=1.0)
tr.detrend("demean")

simplearr = np.ascontiguousarray(tr.data, np.float32)
WinSec = 0.5

N = round(WinSec/tr.stats.delta) + 1

# simplearr = np.array([7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
# simplearr = np.ascontiguousarray(simplearr, np.float32)
# N = 2

myclib.kurtcf.restype = C.c_int
myclib.kurtcf.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                          C.c_int, C.c_int,
                          # OUT
                          np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]
kurt_cf_arr = np.zeros(simplearr.size, dtype=np.float32, order="C")

ret = myclib.kurtcf(simplearr, simplearr.size, N, kurt_cf_arr)
if ret != 0:
    raise MemoryError("Something wrong with AIC picker")


myclib.kurtcf_mean.restype = C.c_int
myclib.kurtcf_mean.argtypes = [np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS'),
                          C.c_int, C.c_int,
                          # OUT
                          np.ctypeslib.ndpointer(
                                        dtype=np.float32, ndim=1,
                                        flags='C_CONTIGUOUS')]
kurt_cf_arr_mean = np.zeros(simplearr.size, dtype=np.float32, order="C")

ret = myclib.kurtcf_mean(simplearr, simplearr.size, N, kurt_cf_arr_mean)
if ret != 0:
    raise MemoryError("Something wrong with AIC picker")


####
print(len(kurt_cf_arr), kurt_cf_arr)
kurt_cf_arr[0:N]=np.nan
plt.plot(kurt_cf_arr,'k')
kurt_cf_arr_mean[0:N]=np.nan
plt.plot(kurt_cf_arr_mean,'r')
plt.show()

print("arrivato")










# myclib.skewcf.restype = C.c_int
# myclib.skewcf.argtypes = [np.ctypeslib.ndpointer(
#                                         dtype=np.float32, ndim=1,
#                                         flags='C_CONTIGUOUS'),
#                           C.c_int, C.c_int,
#                           # OUT
#                           np.ctypeslib.ndpointer(
#                                         dtype=np.float32, ndim=1,
#                                         flags='C_CONTIGUOUS')]
# skew_cf_arr = np.zeros(simplearr.size, dtype=np.float32, order="C")

# ret = myclib.skewcf(simplearr, simplearr.size, N, skew_cf_arr)
# if ret != 0:
#     raise MemoryError("Something wrong with AIC picker")


# myclib.skewcf_mean.restype = C.c_int
# myclib.skewcf_mean.argtypes = [np.ctypeslib.ndpointer(
#                                         dtype=np.float32, ndim=1,
#                                         flags='C_CONTIGUOUS'),
#                           C.c_int, C.c_int,
#                           # OUT
#                           np.ctypeslib.ndpointer(
#                                         dtype=np.float32, ndim=1,
#                                         flags='C_CONTIGUOUS')]
# skew_cf_arr_mean = np.zeros(simplearr.size, dtype=np.float32, order="C")

# ret = myclib.skewcf_mean(simplearr, simplearr.size, N, skew_cf_arr_mean)
# if ret != 0:
#     raise MemoryError("Something wrong with AIC picker")




# ####
# print(len(kurt_cf_arr), kurt_cf_arr)
# kurt_cf_arr[0:N]=np.nan
# plt.plot(skew_cf_arr,'k')
# kurt_cf_arr_mean[0:N]=np.nan
# plt.plot(skew_cf_arr_mean,'r')
# plt.show()

# print("arrivato")


