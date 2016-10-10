import numpy as np
import os, sys, time
from matplotlib import pyplot as plt
from . import common, kfit

def get_geophone_constants():
    """
    :return: A dictionary of constants used in the rest of this module
    """
    const = {'Q': 2.0,
             'f0': 4.5,
             'Z12': 33.1,
             'RT': 380,
             'LT': 0.139,
             'm0': 23.00e-3,
             'Z_i': 1e6,
             'R_S': 1e4}

    return const

def get_geophone_displacement(f, V, Q=None, f0=None, Z12=None):
    """
    :param f: frequency points
    :param V: Voltage as measured by the geophone, array must be the same length as f
    :param Q: Q factor, if not specified, it will be taken from the list of constants
    :param f0: Resonance frequency in Hz
    :param Z12: Impedance
    :return: displacement in m, array of the same length as f & V
    """

    sensitivity = np.abs(get_geophone_sensitivity(f, Q=Q, f0=f0, Z12=Z12))
    velocity = V / sensitivity
    displacement = velocity / (2 * np.pi * f)

    return displacement

def get_geophone_sensitivity(f, Q=None, f0=None, Z12=None):
    """
    :param f: frequency in Hz, array like
    :param Q: Q of the geophone
    :param f0: Resonance frequency in Hz
    :param Z12: Sensitivity
    :return: Transfer function in V/(m/s) for each frequency point.
    """
    const = get_geophone_constants()
    if Q is None:
        Q = const['Q']
    if f0 is None:
        f0 = const['f0']
    if Z12 is None:
        Z12 = const['Z12']

    def H(f, f0, Q):
        x = f / np.float(f0)
        return x ** 2 / (1 - x ** 2 + 1j * x / np.float(Q))

    return H(f, f0, Q) * Z12

def geophone_func(x, Q, f0, Z12, RT, LT):
    """
    :param p: array of fit parameters, in order: [Q, f0, Z12, RT, LT]
    :param x: array with frequency points
    :return: rho, voltage divider signal: Vout/Vin.
    """
    const = get_geophone_constants()
    m0 = const['m0']
    Zi = const['Z_i']
    R_S = const['R_S']

    # Parameters to be fitted:
    # Q, f0, Z12, RT, LT = p

    def Z_E(f):
        s = 2 * np.pi * f
        y = f / np.float(f0)
        return RT + 1j * s * LT + 1j * s * Z12 ** 2 / (m0 * (2 * np.pi * f0) ** 2 * (1 - y ** 2 + 1j * y / Q))

    def Z_E_prime(f, Z_i):
        return Z_E(f) * Z_i / (Z_E(f) + Z_i)

    def Z_S_prime(R_S, Z_i):
        return R_S * Z_i / (R_S + Z_i)

    def rho(f):
        return Z_E_prime(f, Zi) / (Z_E_prime(f, Zi) + Z_S_prime(R_S, Zi))

    return np.abs(rho(x))

def fit_calibration_curve(xdata, ydata, init_guess, **kwarg):
    """
    :param xdata: frequency points
    :param ydata: voltage divider output rho(f)
    :param init_guess: list of the form: [Q, f0, Z12, RT, LT]
    :return:

    Optional parameters: domain = [xstart, xstop], showfit = Bool, showstartfit = Bool,
    showdata = Bool, label = '', mark_data = '', mark_fit = ''
    """
    bestfitparams, fitparam_errors = kfit.fitbetter(xdata, ydata, geophone_func, init_guess, show_diagnostics=True,
                                                     **kwarg)

    print("Fit results with 1 sigma:")
    params = ['Q', 'f0', 'Z12', 'RT', 'LT']
    for k in range(5):
        print("{} = {} +/- {}".format(params[k], bestfitparams[k], fitparam_errors[k]))

    return bestfitparams, fitparam_errors

def get_geophone_spectrum(df, G, freqlim=[1.0, 200.0], Q=1.54, f0=4.55, Z12=31.58, do_imshow=True, do_plot=True,
                          ret=False, name=None, do_meters_per_sqrt_Hz=False):
    """
    :param df: filepath of the datafile
    :param G: gain of the amplifier
    :param freqlim: [fmin, fmax]
    :param Q: quality factor from calibration
    :param f0: resonance frequency from calibration
    :param Z12: resonance frequency from calibration
    :param do_imshow: show a color plot of all repetitions
    :param do_plot: show the mean of all the repetitions
    :param ret: True/False to get mean
    :return:
    """
    data = dataCacheProxy(expInst='alazar_scope', filepath=os.path.join(df))

    t = data.get('t')
    ch1 = data.get('ch1')

    if np.sum(ch1) == 0:
        print("WARNING: sum of ch1 is 0 for %s" % df)

    psd = np.zeros((np.shape(ch1)[0], np.shape(ch1)[1] / 2))
    rms = list()
    for idx in range(np.shape(ch1)[0]):
        freq, spectral_density = common.plot_spectrum(ch1[idx, :] / np.float(G), t[idx, :], freqlim='auto',
                                                      linear=False,
                                                      type='psd', do_plot=False, verbose=False)

        psd[idx, :] = np.abs(spectral_density)

        start = np.where(freq > freqlim[0])[0][0]
        stop = np.where(freq < freqlim[1])[0][-1]
        rms.append(get_frequency_rms(
            get_geophone_displacement(freq[start:stop], 2 * psd[idx, :][start:stop], Q=Q, f0=f0, Z12=Z12)))

    start = np.where(freq > freqlim[0])[0][0]
    stop = np.where(freq < freqlim[1])[0][-1]

    if do_meters_per_sqrt_Hz:
        meanpsd = 2 * np.mean(psd, axis=0) * np.sqrt(max(t[0, :]))
    else:
        meanpsd = 2 * np.mean(psd, axis=0)

    if do_imshow:
        fig = plt.figure(figsize=(12., 4.))
        plt.subplot(111)
        common.configure_axes(13)
        plt.imshow(20 * np.log10(psd), interpolation='none', aspect='auto',
                   extent=[np.min(freq), np.max(freq), np.shape(psd)[0], 1])
        plt.xlabel('FFT frequency (Hz)')
        plt.ylabel('Repetition #')
        plt.clim([-140, -90])
        plt.colorbar()
        plt.xlim(freqlim)

    calibrated_displacement = get_geophone_displacement(freq[start:stop], meanpsd[start:stop], Q=Q, f0=f0, Z12=Z12)

    if do_plot:
        fig2 = plt.figure(figsize=(12., 4.))
        common.configure_axes(13)
        plt.plot(freq[start:stop], calibrated_displacement, '-r')
        plt.xlabel('FFT frequency (Hz)')
        plt.yscale('log')
        plt.xlim(freqlim)

    if not do_meters_per_sqrt_Hz:
        print("RMS value of %s between %.2f Hz and %.2f Hz is %.3e +/- %.1e m" % (
            name, freqlim[0], freqlim[1], get_frequency_rms(calibrated_displacement), np.std(rms)))
        plt.ylabel(r'Calibrated displacement (m)')
    else:
        print("RMS value of %s between %.2f Hz and %.2f Hz is %.3e +/- %.1e m" % (
            name, freqlim[0], freqlim[1], get_frequency_rms(calibrated_displacement) / np.sqrt(np.max(t[0, :])),
            np.std(rms) / np.sqrt(np.max(t[0, :]))))
        plt.ylabel(r'Calibrated displacement ($\mathrm{m}/\sqrt{\mathrm{Hz}}$)')

    if ret:
        return freq[start:stop], calibrated_displacement

def compare_traces(dfs, gains, freqlim, Qs=1.54, f0s=4.552, Z12s=31.58, leg=None, ylim=None, psd_units=True):
    """
    Compare traces side by side in a figure.
    :param dfs: a list of filepaths
    :param gains: float, or list of floats with same length as dfs
    :param freqlim: list [fmin, fmax]
    :param Qs: float, or list of floats with same length as dfs
    :param f0s: float, or list of floats with same length as dfs
    :param Z12s: float, or list of floats with same length as dfs
    :param leg: list of string containing labels for the legend
    :param ylim: limits for the y-axis
    :param psd_units: True/False
    :return: None
    """
    from mpltools import color

    fig = plt.figure(figsize=(12., 4.))
    common.configure_axes(13)
    color.cycle_cmap(len(dfs), cmap=plt.cm.jet)

    for i, df in enumerate(dfs):
        try:
            string = leg[i]
        except:
            string = ''

        if isinstance(Qs, (float)):
            Q = Qs
        elif isinstance(Qs, (list, np.ndarray)) and len(Qs) == len(dfs):
            Q = Qs[i]
        else:
            print("Qs must have the same length as dfs or must be a float.")

        if isinstance(f0s, (float)):
            f0 = f0s
        elif isinstance(f0s, (list, np.ndarray)) and len(f0s) == len(dfs):
            f0 = f0s[i]
        else:
            print("f0s must have the same length as dfs or must be a float.")

        if isinstance(Z12s, float):
            Z12 = Z12s
        elif isinstance(Z12s, (list, np.ndarray)) and len(Z12s) == len(dfs):
            Z12 = Z12s[i]
        else:
            print("Z12s must have the same length as dfs or must be a float.")

        if isinstance(gains, (float)):
            G = gains
        elif isinstance(gains, (list, np.ndarray)) and len(gains) == len(dfs):
            G = gains[i]
        else:
            print("f0s must have the same length as dfs or must be a float.")

        f, cal = get_geophone_spectrum(df, G, freqlim=freqlim, Q=Q, f0=f0, Z12=Z12, do_imshow=False,
                                       do_plot=False, ret=True, name=leg[i], do_meters_per_sqrt_Hz=psd_units)
        plt.plot(f, cal, label=string)

    plt.xlabel('FFT frequency (Hz)')
    plt.yscale('log')
    plt.xlim(freqlim)
    fig.patch.set_facecolor('white')

    if leg is not None:
        common.legend_outside(prop={'size': 10})
    if ylim is not None:
        plt.ylim(ylim)

def subtract_traces(dfs, gains, freqlim, Qs=1.54, f0s=4.552, Z12s=31.58, leg=None, ylim=None, psd_units=True):
    """
    dfs: a list of filepaths. Subtract second from the 1st file.
    gains: a float
    freqlim: list [fmin, fmax]
    Q, f0, Z12 optional
    """
    from mpltools import color

    fig = plt.figure(figsize=(12., 4.))
    common.configure_axes(13)
    color.cycle_cmap(len(dfs), cmap=plt.cm.jet)

    for i, df in enumerate(dfs):
        try:
            string = leg[i]
        except:
            string = ''

        if isinstance(Qs, (float)):
            Q = Qs
        elif isinstance(Qs, (list, np.ndarray)) and len(Qs) == len(dfs):
            Q = Qs[i]
        else:
            print("Qs must have the same length as dfs or must be a float.")

        if isinstance(f0s, (float)):
            f0 = f0s
        elif isinstance(f0s, (list, np.ndarray)) and len(f0s) == len(dfs):
            f0 = f0s[i]
        else:
            print("f0s must have the same length as dfs or must be a float.")

        if isinstance(Z12s, (float)):
            Z12 = Z12s
        elif isinstance(Z12s, (list, np.ndarray)) and len(Z12s) == len(dfs):
            Z12 = Z12s[i]
        else:
            print("Z12s must have the same length as dfs or must be a float.")

        if isinstance(gains, (float)):
            G = gains
        elif isinstance(gains, (list, np.ndarray)) and len(gains) == len(dfs):
            G = gains[i]
        else:
            print("f0s must have the same length as dfs or must be a float.")

        if i == 0:
            f, cal0 = get_geophone_spectrum(df, G, freqlim=freqlim, Q=Q, f0=f0, Z12=Z12, do_imshow=False,
                                       do_plot=False, ret=True, name=leg[i], do_meters_per_sqrt_Hz=psd_units)
        else:
            f, cal = get_geophone_spectrum(df, G, freqlim=freqlim, Q=Q, f0=f0, Z12=Z12, do_imshow=False,
                                           do_plot=False, ret=True, name=leg[i], do_meters_per_sqrt_Hz=psd_units)
            plt.plot(f, cal-cal0, label=string)

    plt.xlabel('FFT frequency (Hz)')
    plt.yscale('log')
    plt.xlim(freqlim)
    fig.patch.set_facecolor('white')

    if leg is not None:
        common.legend_outside(prop={'size': 10})
    if ylim is not None:
        plt.ylim(ylim)

def get_frequency_rms(fft):
    """
    There is a factor of sqrt(2) because the negative frequencies aren't taken into account.
    Input should be a fourier transform, not a power spectral density, or amplitude spectral density.
    """
    return np.sqrt(np.trapz(np.abs(np.sqrt(2) * fft) ** 2))

def process_calibration_measurement(df_vout, df_vin, fit_domain=[0.5, 100]):
    """
    Fit the calibration measurement. Requires 2 input files, One containing the Vout and one containing the Vin.
    These are the 2 series of voltages measured after and before the 10k resistor.
    :param df_vout: File path of file containing output voltage as function of frequency
    :param df_vin: File path of file containging input voltage as function of frequency
    :param fit_domain: List of [fmin, fmax]
    :return: Fit results, Fit errors
    """
    # Vout measurement:
    data = dataCacheProxy(expInst='geophone_calibration', filepath=df_vout)
    fout = data.get('f')
    Vout = data.get('meanV')
    Vout_err = data.get('stdV')

    # Vin measurement:
    data = dataCacheProxy(expInst='geophone_calibration', filepath=df_vin)
    fin = data.get('f')
    Vin = data.get('meanV')
    Vin_err = data.get('stdV')

    sigma_rho = np.sqrt((1 / Vin) ** 2 * Vout_err ** 2 + (Vout / Vin ** 2) ** 2 * Vin_err ** 2)

    try:
        fr, err_dict = fit_calibration_curve(np.array(fin, dtype=np.float64),
                                             np.array(Vout / Vin, dtype=np.float64),
                                             [2.0, 4.5, 30.0, 570.0, 0.139], showfit=False,
                                             showstartfit=False, domain=fit_domain)
        success = True
    except RuntimeError:
        success = False
        print("Error in fitting")

    fplot = np.logspace(-1, 2, 1E3)
    plt.figure(figsize=(6., 4.))
    common.configure_axes(13)
    plt.errorbar(fin, Vout / Vin, yerr=sigma_rho, fmt='o', color='r')
    if success:
        plt.plot(fplot, geophone_func(fplot, *fr), '-k', lw=2.0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Vout/Vin (V)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(np.min(Vout / Vin) / 1.5, np.max(Vout / Vin) * 1.5)
    plt.xlim(min(fin), max(fin))

    return fr, err_dict