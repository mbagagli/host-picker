# HOST picker

**Author:** _Matteo Bagagli_
**Date:** _08/2019 @ ETH-Zurich_
**Version:** _1.0.1_

## Introduction
The HOST picker is a High-Order-STatistics based picker.
This algorithm is a variation of the one described in [Baillard et al. 2013](http://www.ipgp.fr/~mangeney/Baillard_etal_bssa_2014),
and it is convenient for multiple usage and different scientific problem.
This picker is yet simple and stable, but powerful in detecting consistent phase arrival (or general transient).

## Installation
To install the picker, simply run on an open terminal:
```
$ pip install DIR_WHERE_HOST_IS
```

and you're ready to go. Just note that you need to have installed all the required package listed in the `requirements.txt` file.
To check that everything is up and running type:
```
$ pytest
```
If all test passed then you're ready to go, you now can import the modules in your scripts/projects...enjoy!

