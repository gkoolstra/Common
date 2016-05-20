import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import kfit, cmath
from tabulate import tabulate

def get_phase(array):
    """
    Returns the phase of a complex array.
    :param array: Array filled with complex numbers
    :return: Array
    """
    phase_out = np.zeros([len(array)])
    for idx,k in enumerate(array):
        phase_out[idx] = cmath.phase(k)
    return phase_out

def remove_offset(xdata, ydata, f0=None):
    """
    Removes a constant offset and places ydata at point f0 at 0.
    :param xdata: fpoints.
    :param ydata: ydata.
    :param f0: point at which ydata will be set to 0. if f0=None it will be the center of the array.
    :return: the ydata with the offset removed.
    """
    if f0 is None:
        f0_idx = len(xdata)/2
    else:
        f0_idx = find_nearest(xdata, f0)
    y_f0 = ydata[f0_idx]
    return ydata - y_f0

def remove_slope(xdata, ydata):
    """
    Removes a slope from data using the first and last points of xdata/ydata to determine the slope.
    :param xdata: fpoints.
    :param ydata: ydata.
    :return: ydata with the slope removed.
    """
    return ydata-(ydata[-1]-ydata[0])/(xdata[-1]-xdata[0])*xdata

def recenter_phase(xdata, ydata, f0):
    """
    Removes the slope and phase offset such that the phase is flat and has a 0 at f0.
    :param xdata: fpoints
    :param ydata: phase
    :param f0: Point at which the phase is 0.
    :return:
    """
    phase = remove_slope(xdata, ydata)
    return remove_offset(xdata, phase, f0)

def find_nearest(array, value):
    """
    Finds the nearest value in array. Returns index of array for which this is true.
    """
    idx=(np.abs(array-value)).argmin()
    return idx

def moving_average(interval, window_size):
    """
    Outputs the moving average of a function interval over a window_size. Output is 
    the same size as the input. 
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def dBm_to_W(Pdbm):
    """
    Convert power in dBm to power in W
    :param Pdbm: Power in dBm.
    :return: Power in W
    """
    return 10**(Pdbm/10.) * 1E-3

def dBm_to_vrms(Pdbm, Z0=50.):
    """
    Convert power in dBm to rms voltage
    :param Pdbm: Power in dBm.
    :param Z0: Characteristic impedance
    :return: Vrms
    """
    Pmw = 10**(Pdbm/10.)
    return np.sqrt(Pmw*Z0/1E3)

def dBm_to_vpp(Pdbm, Z0=50.):
    """
    Convert power in dbm to peak-to-peak voltage given certain characteristic impedance Z0
    :param Pdbm: Power in dBm
    :param Z0: Characteristic impedance
    :return: Vpp
    """
    Pmw = 10**(Pdbm/10.)
    return 2 * np.sqrt(2) * np.sqrt(Pmw*Z0/1E3)

def save_figure(fig, save_path=None, open_explorer=True):
    """
    Saves a figure with handle "fig" in "save_path". save_path does not need to be specified, if not specified
    the function will create a new file in S:\Gerwin\iPython notebooks\Figures under the current date.
    :param fig: Figure handle
    :param save_path: Filename for the file to be saved. May be None
    :param open_explorer: Open a process of windows explorer showing the file, for easy copy & paste into slides
    :return: Nothing
    """
    import subprocess, time, os

    # Check if path is accessible
    if save_path is None:
        base_path = r"S:\Gerwin\iPython notebooks\Figures"
    else:
        base_path = save_path

    date = time.strftime("%Y%m%d")

    # Create a file name
    file_exists = True
    i = 0
    while file_exists:
        idx = str(1000 + i)
        save_path = os.path.join(base_path, "%s_figure_%s.png"%(date, idx[1:]))
        if not os.path.isfile(save_path):
            break
        else:
            i += 1

    if os.path.exists(os.path.split(save_path)[0]):
        fig.savefig(save_path, dpi=300)

        time.sleep(1)
        if open_explorer:
            subprocess.Popen(r'explorer /select,"%s"'%save_path)

    else:
        print "Desired path %s does not exist."%(save_path)


def mapped_color_plot(xdata, ydata, cmap=plt.cm.viridis, clim=None, type='x', log_scaling=False):
    """
    Plot points in a data set with different color. The value of the color is determined either by the x-value or the
    y-value and can be scaled linearly or logarithmically.
    :param xdata: x-points
    :param ydata: y-points
    :param cmap: plt.cm instance
    :param clim: Tuple (cmin, cmax). Default is None.
    :param type: Either 'x' or 'y'
    :param log_scaling: Scale data logarithmically
    :return:
    """
    if clim is None:
        if type == 'x':
            if log_scaling:
                vmin, vmax = np.min(np.log10(xdata)), np.max(np.log10(xdata))
            else:
                vmin, vmax = np.min(xdata), np.max(xdata)
        else:
            if log_scaling:
                vmin, vmax = np.min(np.log10(xdata)), np.max(np.log10(xdata))
            else:
                vmin, vmax = np.min(xdata), np.max(xdata)
    else:
        vmin, vmax = clim

    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for x, y in zip(xdata, ydata):
        if type == 'x':
            if log_scaling:
                plt.plot(x, y, 'o', m.to_rgba(np.log10(x)))
            else:
                plt.plot(x, y, 'o', m.to_rgba(x))
        else:
            if log_scaling:
                plt.plot(x, y, 'o', m.to_rgba(np.log10(y)))
            else:
                plt.plot(x, y, 'o', m.to_rgba(y))



def configure_axes(fontsize):
    """
    Creates axes in the Arial font with fontsize as specified by the user
    :param fontsize: Font size in points, used for the axes labels and ticks.
    :return: None
    """
    import platform
    if platform.system() == 'Linux':
        matplotlib.rc('font', **{'size': fontsize})
        matplotlib.rc('figure', **{'dpi': 80, 'figsize': (6.,4.)})
    else:
        matplotlib.rc('font', **{'size': fontsize, 'family':'sans-serif','sans-serif':['Arial']})
        matplotlib.rc('figure', **{'dpi': 80, 'figsize': (6.,4.)})


def legend_outside(**kwargs):
    """
    Places the legend outside the figure area. Useful for a busy figure where the legend overshadows data.
    :param kwargs: Extra arguments
    :return: None
    """
    ax = plt.gca()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **kwargs)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

def plot_opt(color, style='transparent', msize=8):
    """
    Usage: plot(x, y, **plot_opt('red'))
    :param color: String denoting the color of the plot
    :param style: Select 'transparent' for transparent plotting style.
    :param msize: Marker size in points. Defaults to 8 if not specified.
    :return: Dictionary of marker styles.
    """
    if style == 'transparent':
        return {'marker': 'o', 'ms':msize, 'mew':1, 'mfc': color ,'mec': color, 'alpha' : 0.4}
    else:
        return {'marker': 'o', 'ms':6, 'mew':2, 'mec': color ,'mfc': 'white'}

def setup_twinax(color1='black', color2='black'):
    """
    Sets up a double axes plot (two y-axes, one x-axis). It also colors the axes according
    to color1 (left) and color2 (right). Returns the left and right axes.
    :param color1: String denoting the color of the left axis
    :param color2: String denoting the color of the right axis
    :return: 2 axis handles
    """
    ax = plt.gca()
    ax2 = ax.twinx()

    ax2.tick_params(axis='y', colors=color2)
    ax2.yaxis.label.set_color(color2)
    ax.tick_params(axis='y', colors=color1)
    ax.yaxis.label.set_color(color1)

    return ax, ax2

def plot_spectrum(y, t, ret=True, do_plot=True, freqlim='auto', ylim='auto', logscale=True, linear=True, type=None,
                  verbose=True, do_phase=False):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t). t should have evenly spaced entries, i.e. constant dt.
    Returns the spectrum.

    Example usage:

    t = arange(0,1,0.01)  # time vector
    y = sin(2*pi*t)
    plot_spectrum(y,t)

    Should produce a peak at 1 Hz.

    :param y: Amplitude in V.
    :param t: Time in seconds.
    :param ret: True/False. Default is True. Set to false if returning spectral data is not desired.
    :param do_plot: True/False. Plot the spectrum.
    :param freqlim: Tuple. Limits of the frequency axis.
    :param ylim: Tuple. Limits of y-axis.
    :param logscale: True/False for logarithmic y axis.
    :param linear: True/False for plotting in dB. When linear=False, automatically turns logscale to False
                   (negative dB cannot be converted to logscale)
    :param type: 'psd' or 'asd'. Default is 'asd'
    :param verbose: True/False, Prints the maximum contribution in the spectrum.
    :param do_phase: True/False. Plots a second figure with the Phase spectral density.
    :return: Frequency, ASD (complex) for type='asd', Frequency, PSD (real) for type='psd'
    """
    
    if len(t) < 2: 
        raise ValueError("Length of t is smaler than 2, cannot compute FFT.")
    else:
        dt = t[1]-t[0]
        n = len(y) # length of the signal
        T = n*dt
        frq = np.arange(n)/float(T) # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        Y = np.fft.fft(y)/float(n) # fft computing and normalization
        Y = Y[range(n/2)] # maps the negative frequencies on the positive ones. Only works for real input signals!

        if do_plot:
            plt.figure(figsize = (6.,4.))
            ax1 = plt.subplot(111)
            configure_axes(13)
            if linear:
                if type == 'psd':
                    ax1.plot(frq, np.abs(Y)**2, 'r')
                    ax1.set_ylabel(r'Power spectral density (V$^2$)')
                else:
                    ax1.plot(frq, np.abs(Y), 'r') # plotting the spectrum
                    ax1.set_ylabel(r'Amplitude spectral density (V)')
                scientific_axes = 'both'
            else:
                if type == 'psd':
                    ax1.plot(frq, 20*np.log10(np.abs(Y)), 'r')
                    ax1.set_ylabel(r'Power spectral density (V$^2$)')
                else:
                    ax1.plot(frq, 10*np.log10(np.abs(Y)), 'r')
                    ax1.set_ylabel(r'Amplitude spectral density (V)')

                logscale = False
                scientific_axes = 'x'

            ax1.ticklabel_format(style='sci', axis=scientific_axes, scilimits=(0,0))

            if do_phase:
                plt.figure(figsize=(6.,4.))
                ax2 = plt.subplot(111)
                ax2.plot(frq, get_phase(Y), 'lightgreen')
                ax2.set_ylabel('Phase density (rad)')
                ax2.set_xlabel('$f$ (Hz)')

            ax1.set_xlabel('$f$ (Hz)')

            if logscale:
                ax1.set_yscale('log')

            if freqlim != 'auto':
                try:
                    ax1.set_xlim(freqlim)
                    if do_phase:
                        ax2.set_xlim(freqlim)
                except:
                    print "Not a valid xlim, please specify as [min, max]"

            if ylim != 'auto':
                try:
                    ax1.set_ylim(ylim)
                    if do_phase:
                        ax2.set_ylim(ylim)
                except:
                    print "Not a valid ylim, please specify as [min, max]"


        if verbose:
            print "Maximum contribution to signal is %.2e for a frequency of %.2e Hz"\
                    %(np.max(np.abs(Y)), frq[np.where(np.max(np.abs(Y)))[0]])
    
        if ret:
            if type!='psd':
                return frq,Y
            else:
                return frq, np.abs(Y)**2

def split_power(power_in, conversion_loss):
    """
    Calculates the power at the output of a power splitter.
    :param power_in: power at the input in dBm
    :param conversion_loss:
    :return:
    """
    power_in_dB = power_in-conversion_loss
    inW = 0.5*10**(power_in_dB/10.)
    power_out_dB = 10*np.log10(inW)
    return power_out_dB

def fit_lorentzian(g_A, g_gamma, g_x0, *arg):
    """
    Fit lorentzian lineshape. Parameters:
    A: scaled amplitude
    gamma: half width
    x0: resonance frequency
    """
    fitfunc_str = '1/pi A * gamma / ((x-x0)^2 + gamma^2)'

    A = fit.Parameter(g_A, 'A')
    gamma = fit.Parameter(g_gamma, 'gamma')
    x0 = fit.Parameter(g_x0, 'x0')

    p0 = [A, gamma, x0]

    def fitfunc(x):
        return A()/((x-x0())**2+gamma()**2)

    return p0, fitfunc, fitfunc_str

def fit_hanger(g_f0, g_Qi, g_Qc, g_df, g_scale):
    """
    Fit hanger function with asymmetric lineshape. 
    Asymmetry is due to different load impedance (> or < 50 Ohms)
    and is represented by parameter df (note: in Hz)
    """
    fitfunc_str = 'Asymmetric hanger function'

    f0 = fit.Parameter(g_f0, 'f0')
    Qi = fit.Parameter(g_Qi, 'Qi')
    Qc = fit.Parameter(g_Qc, 'Qc')
    df = fit.Parameter(g_df, 'df')
    scale = fit.Parameter(g_scale, 'scale')

    p0 = [f0, Qi, Qc, df, scale]

    def fitfunc(x):
        a=(x-(f0()+df()))/(f0()+df())
        b=2*df()/f0()
        Q0=1./(1./Qi()+1./Qc())
        return scale()*(-2.*Q0*Qc() + Qc()**2. + Q0**2.*(1. + Qc()**2.*(2.*a + b)**2.))/(Qc()**2*(1. + 4.*Q0**2.*a**2.))

    return p0, fitfunc, fitfunc_str

def fit_kinetic_fraction(g_alpha, g_f0, g_Tc):
    """
    Fit kinetic induction from temperature sweep of the resonance frequency
    """
    #fitfunc_str = 'f0 * 1/sqrt(1 + alpha * 1/(1-T/Tc))'
    fitfunc_str = 'f0*(1-alpha/2*(1-(T/Tc)**4)**-1)'

    f0 = fit.Parameter(g_f0, 'f0')
    alpha = fit.Parameter(g_alpha, 'alpha')
    Tc = fit.Parameter(g_Tc, 'Tc')
    
    p0 = [f0, alpha, Tc]

    def fitfunc(x):
        #return f0()*1/np.sqrt(1+alpha()/(1-x**4))
        return f0()*(1-alpha()/2.*1/(1-(x/Tc())**4))

    return p0, fitfunc, fitfunc_str    

def Qext(L, C, Cin, Cout):
    """
    Calculates the external Q value of the resonator from the values 
    of the coupling capacitors, and the specifications of the resonator. 
    """
    w0 = 1/np.sqrt(L*C)
    Z = np.sqrt(L/C)
    Z0 = 50
    print "Resonance frequency = f0 = %.3e Hz"%(w0/(2*np.pi))
    print "Impedance = Z = %.3e Ohms"%Z
    return 1/(Z0*w0**3*L*(Cin**2+Cout**2))

def CfromQ(L, C, Qext):
    """
    Calculates the coupling capacitance for a desired external Q value. 
    Input parameters are L and C values of the resonator. 
    NOTE: This assumes that both coupling capacitors (Cin and Cout) are the same. 
    """
    w0 = 1/np.sqrt(L*C)
    Z = np.sqrt(L/C)
    Z0 = 50
    print "Resonance frequency = f0 = %.3e Hz"%(w0/(2*np.pi))
    print "Impedance = Z = %.3e Ohms"%Z
    return 1/np.sqrt(2*Qext*Z0*w0**3*L)

def beep():
    """
    Notifies the user of a certain event by playing a sound sequence.
    This sound sequence is a C E G C chord. :)
    :return: None
    """
    for k in range(4):
        winsound.Beep(261, 250)
        winsound.Beep(330, 250)
        winsound.Beep(392, 250)
        winsound.Beep(523, 250)

        winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS)

def simple_beep():
    """
    Play a simple windows sound.
    :return: None
    """
    winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS)

def q_finder(magnitudes, fpoints, debug=False, start_idx=None):
    """
    Method to find the Q factor of a frequency vs magnitude (dB) trace.
    :param magnitudes: Magnitude of the transmission peak in dB
    :param fpoints: Frequency in Hz
    :param debug: Prints the magnitude at the current time step
    :param start_idx: Point at which the searching for the -3dB points should begin. Default is the maximum of the curve.
    :return: Q
    """
    if start_idx is None: 
        start_idx = np.where(magnitudes == np.max(magnitudes))[0][0]
    else: 
        pass 

    max_mag = magnitudes[start_idx]
    if debug: print "Maximum magnitude is %.2f"%max_mag
    f0 = fpoints[start_idx]
    walking_magnitude = max_mag
    k=1

    while abs(walking_magnitude-max_mag) < 3:
        walking_magnitude = magnitudes[start_idx+k]
        if debug: print walking_magnitude
        k+=1

    threedbpointright = fpoints[start_idx+k]
    if debug: print threedbpointright
    k=1

    while abs(walking_magnitude-max_mag) < 3:
        walking_magnitude = magnitudes[start_idx-k]
        k+=1
    
    threedbpointleft = fpoints[start_idx-k]
    if debug: print threedbpointleft

    df = threedbpointright-threedbpointleft
    return f0/df

def get_thermal_photons(f, T):
    """
    Returns the number of thermal photons at frequency f and temperature T.
    :param f: Frequency of the photons
    :param T: Temperature of the bath
    :return:
    """
    kB = 1.38E-23
    h = 6.63E-34
    return (kB*T)/(h*f)

def get_noof_photons_in_cavity(P, f0, Q):
    """
    Returns the number of photons in the cavity on resonance
    :param P: Input power in the drive in dBm
    :param f0: Resonant frequency of the cavity
    :param Q: Q of the cavity
    :return:
    """
    hbar = 1.055E-34
    w0 = 2*np.pi*f0
    P_W = 10**(P/10.)
    kappa = w0/Q
    return P_W/(hbar*w0*kappa/(2*np.pi))

def get_noof_photons_in_input(P, f):
    """
    Returns the number of photons in the input drive for a given power P and frequency f.
    :param P: Input power in the drive in dBm
    :param f: Frequency of the input photons
    :return:
    """
    hbar = 1.055E-34
    w0 = 2*np.pi*f
    P_W = 10**(P/10.)
    return P_W/(hbar*w0**2/(2*np.pi))

def pad_zeros(f, Y, until='auto', verbose=False):
    """
    Fill an array with zeros up to a specific index "until". Until may be "auto" which means the program will
    add zeros until the size of the new array is a power of two.
    :param f: Frequency domain data (1D array)
    :param Y: Amplitude data (1D or 2D array)
    :param until: 'auto' or an integer
    :param verbose: True/False
    :return:
    """
    if len(f) != len(Y):
        print "Expected arrays of same length, got different size arrays."
    else:
        if until == 'auto':
            # Pad zeros until the nearest power of 2:
            Ni = len(Y)
            Nf = int(2**(np.ceil(np.log2(Ni))))
        elif isinstance(until, int):
            Nf = until
        else:
            raise ValueError('Until may be one of two things: auto or an integer.')

        if len(Y.shape) > 1:
            # 2D arrays
            Ynew = np.zeros((Y.shape[0], Nf))
            Ynew[:,:Ni] = Y

        else:
            Ynew = np.zeros(Nf)
            fnew = np.zeros(Nf)
            df = np.diff(f)[0]

            Ynew[:Ni] = Y
            fnew[:Ni] = f
            fnew[Ni:Nf] = np.linspace(f[-1]+df, f[0]+Nf*df, Nf-Ni)

        if verbose:
            try:
                print "New shape of array is (%d x %d)" % (Ynew.shape[0], Ynew.shape[1])
            except:
                print "New length of array is %d" % (Ynew.shape[0])

        return fnew, Ynew

def get_psd(t, y, verbose=False, window=True):
    """
    Computes the periodogram Only for real signals.
    :param t: Time in seconds
    :param y: Amplitude in V
    :param verbose: True/False. This will print the value of the largest contribution to the PSD
    :param window: This will multiply the FFT with a Hanning window to reduce influence from the finite measurement time.
                   The defult is True. The Hanning window's efficiency is largest for finite size time traces.
                   More on Hanning windows here: https://en.wikipedia.org/wiki/Window_function#Hann_.28Hanning.29_window
    :return: frequency, PSD
    """

    if len(t) < 2:
        raise ValueError("Length of t is smaler than 2, cannot compute FFT.")
    else:
        dt = t[1]-t[0]
        n = len(y) # length of the signal
        T = n*dt
        frq = np.arange(n)/float(T) # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        if window:
            y *= np.hanning(n)

        Y = np.fft.fft(y)*np.sqrt(T)/float(n) # fft computing and normalization
        Y = Y[range(n/2)] # maps the negative frequencies on the positive ones. Only works for real input signals!

        if verbose:
            print "Maximum contribution to signal is %.2e for a frequency of %.2e Hz"\
                    %(np.max(np.abs(Y)), frq[np.where(np.max(np.abs(Y)))[0]])

        return frq, np.abs(Y)**2

def get_circular_points(radius, npts, theta_offset=0, do_plot=True):
    """
    Plots npts points in a circular fashion and prints their coordinates. Useful when you're machining a part.
    :param radius: Radius of the pattern
    :param npts: Number of points
    :param theta_offset: Additional rotation may be needed of the points.
    :param do_plot: True/False. Creates a plot of the points.
    :return: None
    """

    plt.figure(figsize=(5.,4.5))
    configure_axes(13)

    x, y = list(), list()
    for n in range(npts):
        xn = radius*np.cos((theta_offset + (n+1) * 360./npts) * np.pi/180.)
        yn = radius*np.sin((theta_offset + (n+1) * 360./npts) * np.pi/180.)

        x.append(xn)
        y.append(yn)

        plt.plot(xn, yn, 'or')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(-1.05*radius, 1.05*radius)
    plt.ylim(-1.05*radius, 1.05*radius)

    t = np.linspace(-180., 180., 100)
    xcircle = radius*np.cos(t * np.pi/180.)
    ycircle = radius*np.sin(t * np.pi/180.)
    plt.plot(xcircle, ycircle, '--r')

    print tabulate(zip(x, y), headers=["x", "y"], tablefmt="rst", floatfmt=".4f", numalign="center", stralign='center')