# Common
Common is a module used to analyze scientific data, plot it in a clear manner. including fitting data etc. The main contents of this module are: 

1. common.py
2. kfit.py

These parts should have no dependencies on modules outside of Common, except modules that can be installed using pip. In particular, there should be no dependencies on the slab module. This was the reason that kfit was split off from slab in the first place.

#### Dependencies
Make sure you have the following packages installed, using `pip install package`, where `package` is one of the following

* tabulate: for printing debugging information and calculation results in table format.
* mpltools: mainly used in making figures look nice.

## common.py
This module contains a lot of commonly used functions for microwave engineering, e.g. converting power in dBm to watts, Vpp and Vrms, calculating the number of photons inside a resonator and more. Moreover, it has some functionality to set up a nice blank canvas for plotting your data, defining a standard marker format for all your plots and quickly creating double axes plots (to name a few). The main point of this module is that it contains a lot of functions that are used in your day-to-day scripts, i.e. they are *common*.

## kfit.py
This module was adapted from `slab.dsfit`. It's better documented than `dsfit.py`, and instead of linear least squares, in `kfit.py` we use non-linear least squares fitting from the `scipy.optimize` module. The advantage is that now there's also  a covariance matrix for the output, which we can use to calculate standard deviations on fitted parameters. Currently there are the following functions available, but any additions are welcome:

1. `fit_lor` - Lorentzian
2. `fit_kinetic_fraction` - Kinetic inductance trace as function of temperature
3. `fit_double_lor` - Superposition of two Lorentzians
4. `fit_N_gauss` - Superposition of N Gaussians
5. `fit_exp` - Exponential decay
6. `fit_pulse_err` - Pulse error function
7. `fit_decaysin` - Decaying sine 
8. `fit_sin` - Sine
9. `fit_gauss` - Gaussian
10. `fit_hanger` - Hanger function 
11. `fit_parabola` - Parabola
12. `fit_s11` - Reflection from microwave resonator, 1 or 2 port
13. `fit_fano` - Transmission through resonator with Fano lineshape
14. `fit_lor_asym` - Asymmetric Lorentzian transmission peak, due to shunt capacitor.
15. `fit_poly` - Arbitrary polynomial

Any additions to `kfit.py` should have fit functions of the following form: `fitfunc(x, *p)`, where `x` is the x-data and `p` is a list containing the fit parameters. An example of a fit function is given below: 

```python
def parabolafunc(x, *p):
    """
    Parabola function
    :param x: x-data
    :param p: [a0, a1, a2] where y = a0 + a1 * (x-a2)**2
    :return: p[0] + p[1]*(x-p[2])**2
    """
    return p[0] + p[1]*(x-p[2])**2
```
