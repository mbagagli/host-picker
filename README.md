# HOST

**Author:** _Matteo Bagagli_
**Date:** _10/2021_
**Version:** _2.1.3_

## What is it?
The **HOST** seismic pickers is a **H**igh-**O**rder-**ST**atistics based picker.
This algorithm is a variation of the one described in [Baillard et al. 2013](http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014),
and it is convenient for multiple usage and different scientific problem.
This picker is yet simple and stable, but powerful in detecting consistent phase arrival (or general transient).

## Installation
In general, the usage of separated COND environment is recommended.
For more information check [their page]()
Since version XXX the picker can be found in PyPI repository. Therefore to install the latest
stable version you may simply type:
```bash
$ pip install host
```
and be ready to go.

If you want to be updated with the latest patches or even contribute to
the project (yes, really your PR are welcome!), fork-me and clone it
to your device:
```bash
$ git clone https://github.com/mbagagli/host host
$ cd host/
$ # conda activate myenv (optional)
$ pip install .
```
...et voila'!

Although the package testing is in CI, you could still install `pytest`
 and still inside the cloned folder type:
```bash
$ pytest
```

## Reference

