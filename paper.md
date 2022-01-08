---
title: 'HOST: A Python package for Higher-Order-STatitics seismic pickers'
tags:
  - Python
  - seismology
  - timeseries
  - picker

authors:
  - name: Matteo Bagagli
    orcid: 0000-0001-9409-8667
    affiliation: 1 , 2

affiliations:
 - name: Institute of Geophysics, Department of Earth Sciences, ETH Zürich, Switzerland
   index: 1
 - name: Istituto Nazionale di Geofisica e Vulcanologia, ONT, Rome, Italy
   index: 2

date: 07 January 2022
bibliography: paper.bib

---

# Introduction

Earthquakes are one of the most devastating natural disasters, and their fast
and reliable detection is the core-challenge of the so called
_early-warning systems_ [@nakamura1988, @espinosaaranda1995]. Being able also to precisely measure
the main seismic phases’ arrival times and consistently assess the observation
error is key to produce reliable hypocenter locations. These hypocenters
locations are collected to  create seismic catalogs that are important
in many fields of seismology (e.g., earthquake-hazard and risk mitigation).

At the base of this detection chain, an important role is given by the experienced
seismologists that carefully check the waveform's data-stream for possible seismic arrivals.
This procedure is called _seismic phase-picking_ and the measurements are usually called _picks_.
The manual approach, though, becomes unfeasible when dealing with a large amount
of data: not only in terms of time, but also in terms of inconsistencies due to human-errors.
The usage of _automatic pickers_ (AP) instead, ease the work-load of routine operators and represents the only feasible approach for real-time picking analysis.
In fact, they provide the necessary detection's consistency as they would obviously behave the
same way with the same waveforms. Nonetheless, they could still detect erroneous or
even fake transients.
For these reason, different AP algorithms use different _characteristic functions_ (CF)
to manipulate the waveform data and extract useful information for the detection.
Thus, finding the correct algorithm and its proper parameter tuning is as crucial step for the
success of phase-picking analysis.

Several AP have been developed in the past. There is a widely
used nomenclature to group these algorithms:
[i] Energy-based pickers (e.g., @allen1982,  @baer1987),
[ii] Higher order statistic pickers (e.g., @kuperkoch2010, @baillard2014),
[iii] Autoregressive pickers (e.g., @maeda1985, @sleeman1999),
[iv] Neural network methods (e.g., @zhu2019),

Every algorithm has its own strengths and weaknesses. Describing them is beyond the scope
of this paper, but a complete review can be found in @kuperkoch2012.

# Statement of need

As the name suggests, the **HOST** package here presented belongs to the
Higher-Order-Statistic pickers group. It originally took inspiration from the one proposed
by @baillard2004, although it presents several improvements in terms of
usage and customization possibilities.

The package object-oriented nature, deviates from the canonical function-oriented flows
found in most picking-algorithms packages, where usually is requested to the user to work in a closed
and dedicated environment (i.e., with specific formats and pre-defined, fixed workflows)

The `host-picker`, instead, delivers simple API classes that are easily embeddable in
existent frameworks or routines.
This package keeps the users as the main builder of its own workflow,
providing complete and essential tools for their pipelines.
Considering the complexity of seismic-waveforms, this idea represents an advantage to approach the picking problem.
As stated before, the AP are necessary  when dealing with
large  and highly-heterogeneous datasets. The "all-in-one" solutions, in such cases, are not recommended.
Rather, one should be able to separate and adapt the workflows for different signal analysis.
This is the ultimate aim of this library: to ease the creation of customized picking pipelines and help the seismology community meet their picking study needs.


# Implementation

The core-package itself consists in 3 main modules: the `picker.py` one that contains
the main `HOS` picker class, the `plotting.py` one that contains some plot routines and
the `scaffold.py` one containing all the necessary functions to
transforms CFs and define the picking detection methods.
The latter module could be expanded by adding customized functions to meet
user-specific needs. In order to use the new features, the package must be reinstalled and only afterwards these functions could then be added to their pipelines.
The only third-party library required is the `ObsPy` one [@krischer2015]. This library is well maintained and widely used by the seismological community, and is mainly used for the I/O routines.

The backbone CFs implemented in the package are:

- _Skewness_:
the skewness provides information about positive or negative deviations of the distribution density function from the expectation value.
The skewness is defined as follows, using the 3rd central-moment third central moment of a continuous distribution.
$$
S=\frac{E\left[(X-E[X]]^{3}\right.}{E[X-E[X]]^{3 / 2}}=\frac{m_{3}}{m_{2}^{3 / 2}}
$$

- _Kurtosis_:
The kurtosis provides information about only positive deviations of the distribution density function from the expectation value.
The kurtosis is defined as follows, using the 4th central-moment third central moment of a continuous distribution.
$$
K=\frac{E\left[(X-E[X]]^{4}\right.}{E[X-E[X]]^{4 / 2}}=\frac{m_{4}}{m_{2}^{2}}
$$

Both the continuous CFs series will abruptly change their values when a non-gaussian signal (i.e., earthquake onset) is detected over a gaussian distributed process (i.e., seismic noise).
After the calculation of one of these CF, is possible to modify and transform it by adding a sequence of functions to the picking pipelines. In the package's notebooks, all  functions are listed and explained. Multiple combinations allow the user to adjust the transient detections to its specific time-series nature.

Once the (optional) CF's transformation stage is done, the CF is then passed to the detection stage, where the user may choose among different approaches to define the correct seismic phase's arrival time:

- The `aic` method: it will calculate the auto-regressive AIC model over the given CF and will select the sample with the lowest value as the correct time-arrival. This method is more robust for complex yet energic first arrival.
- The `minima` method: this method will simply detect the sample with the lowest value as the correct time-arrival. Although it may seem too simplistic, this method is effective  with simple signal and is the faster among all methods.
- The `gaussian` method: this approach will consider the first-derivative of the given CF as a gaussian process. It will detect the first sample exceeding a given threshold (based on the distribution's standard deviation) as the correct arrival time.

Another difference with common HOS-based picker is the possibility of a multi-picking and multi-window analysis that the HOST package offers. Because both _skewness_ and _kurtosis_ are calculated with singular, fixed time-windows [@saragiotis2002, @baillard2014], the sensitivity and the precision of such an approach could be lower for complex or emergent onsets. For this reason, the package allows multi-frequency analysis to increase the accuracy and the robustness of the pick definition, by using different time-windows (periods) to estimate multiple CFs.

In case of a multi-frequency approach, each CF (either _skewness_ or _kurtosis_) is transformed and picked the same way as the pipeline indicated by the user. The final pool of observations is then passed to a statistical _triage_ stage where, with a jack-knife approach, it discerns among the _valid_ and _outlier_ observations.
Once declared the mean of our picks population $\bar{X}$, it computes the mean $\bar{x}_{i}$ and the bias $\bar{X} - \bar{x}_{i}$ for each subsample consisting of all but the $i_{th}$ element (Eq. \autoref{eq:triage}):

\begin{equation}\label{eq:triage}
  \bar{x}_{i}=\frac{1}{n-1} \sum_{j=1, j \neq i}^{n} x_{j}, \quad i=1, \ldots, n
\end{equation}

An observation is declared an outlier if its replicate’s absolute bias exceeds the standard deviation of all the replicate’s bias distribution. The final pick, in this scenario, is defined as the median of the valid-observations, and the absolute error uncertainty is the time difference among the last and the first valid observation.

If only a single time window is used, the error associated to the pick is given by a signal to noise ratio with user-defined time windows. In order to mitigate spiky traces, the noise value is evaluated as 2 standard deviation of the noise data and the signal value is estimated as the mean value among the absolute maximum and minimum of the signal data.

Finally, the `skewness`, `kurtosis` and `aic` CF functions are written in C language, making their computation a lot faster and suitable for real-time usage. Their implementations are contained in the package `src` sub-folder and distributed within the package as well. The package is also hosted over the PyPI repository, making its installation a lot easier.


# Conclusion

The use of the `host-picker` package is recommended to seismologist who
are searching an adaptive, customizable picking package that allows
fast detection of seismic phases and possible to use for real-time picking analysis.
Seismologists could use this picker on all kind of seismic source types (e.g., earthquakes, explosions, landslides), although these HOS pickers better perform with regional and teleseismic earthquakes signals.
Nonetheless, the highly customization possibility offered in terms of frequency analysis and transformation pipelines  makes this algorithm suitable as well for general-purpose time-series transient detections.
For a complete picking tutorial and customization examples, the reader is referred to the project's jupyter notebooks section.

# Acknowledgements

I acknowledge the Swiss-AlpArray SINERGIA project `CRSII2_154434/1` by Swiss National Science Foundation (SNSF) for the support of this project.


# References

<!-- ##### EDITOR :  [Leonardo Uieda (@leouieda)](https://www.leouieda.com) -->

<!--

### cite

 `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

All the features of the package and their usage across different settings are illustrated using
Jupyter notebooks as hands-on tutorials. I strongly recommend the user to check them out to get use to the
packages API.One of the requirements for such a package are the well-know seismology library OBSPY() used
principally for I/O routines

#outlook

I hope this user-friendly yet customizable seismic picker package results useful
to the community for better detection and dEDEDE.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

