# HOST

**Author:** _Matteo Bagagli_
**Date:** _10/2021_
**Version:** _2.3.1_

## What is it?
The **HOST** acronym stands for **H**igh-**O**rder-**ST**atistics seismic pickers.
This algorithm took inspiration from the one described in [_Baillard et al. 2014_](10.1785/0120120347).
The idea behing this package is to provide user-friendly classes for seismic phase picking with
_skewness_ and _kurtosis_ carachteristic-functions.
Originally applied to seismograms by [_Saragiotis et al._](10.1109/TGRS.2002.800438),
the HOS-CF are able to detect energic transient when the statistical properties of a seismogram
(or time series) change abruptly. These CF are calculated on moving window with fixed window.

Measurements of statistical properties in a moving window are suitable for frequency-specific
(or expected) transient. For this reason, the picker support a multi-window analysis,
in order to collect more information or even increase the quality of detections.
Both _skewness_ and _kurtosis_ are calculated with C-routine under the hood,
making this package fast and suitable also for realtime picking porpouses.


## Standard installation
In general, the usage of virtual env is a good habit for python users.
Although this package's dependencies is not For separated **conda** environment is recommended.
Since version `v2.3.1` the picker can be found in PyPI repository. Therefore to install the latest
stable version you may simply type:
```bash
$ pip install host-picker
```
and be ready to go.

## Developer installation
If you want to be updated with the latest patches or even contribute to
the project (yes, really your PR are welcome!), fork-me and clone it
to your device:
```bash
$ git clone https://github.com/mbagagli/host host
$ cd host/
$ # conda activate myenv (optional)
$ pip install .
```
... et voila'!

## Additional infos
Although the package testing is in [Travis-CI](), you could still install `pytest` and
inside the cloned folder type:
```bash
$ pytest
```
to check everything is fine

The package comes with jupyter-notebooks (under `books` subdir) where you
can test and understand this picking-algorithm.

## Reference
- Baillard, C., Crawford, W.C., Ballu, V., Hibert, C. and Mangeney, A., 2014. An automatic kurtosis‐based P‐and S‐phase picker designed for local seismic networks. Bulletin of the Seismological Society of America, 104(1), pp.394-409.
- Saragiotis, C.D., Hadjileontiadis, L.J. & Panas, S.M., 2002. PAI-S/K: a robust automatic seismic P phase arrival identification scheme, IEEE Trans. Geosci. Remote Sens. 40, 1395–1404.

