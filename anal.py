import numpy as np
import os, common, kfit
from matplotlib import pyplot as plt

try:
    from data_cache import dataCacheProxy
except:
    print "data_cache module could not be loaded"

def dcbias_sweep(df, do_plot=True, fitspan=2E6):
    """
    Plot the change in resonance frequency vs. bias voltage
    :param df: Data filepath of the h5 file.
    :param do_plot: True/False
    :param fitspan: Fitspan for the Lorentzian fit function.
    :return: bias voltages, resonance frequency, Q, frequency jitter, Q jitter
    """
    d = dataCacheProxy(expInst='dc_bias_sweep', filepath=df)

    biasV, meanQs, meanw0s, stdw0s, stdQs = [], [], [], [], []

    plt.figure(figsize=(8.,4.))
    common.configure_axes(13)

    for p, S in enumerate(d.index()):
        print p,
        d.current_stack = S

        biasV.append(d.get('biasV')[0])
        mags = d.get('mags')
        fpoints = d.get('fpoints')[0]

        Qs, w0s = [], []

        for k in range(np.shape(mags)[0]):

            center = fpoints[np.argmax(mags[k,:])]

            try:
                fr = kfit.fitlor(fpoints, common.dBm_to_W(mags[k,:]), verbose=False, showfit=False,
                                  domain=[center-fitspan/2., center+fitspan/2.])

                if do_plot:
                    plt.plot(fpoints, 10*np.log10(common.dBm_to_W(mags[k,:])), '.k')
                    plt.plot(fpoints, 10*np.log10(kfit.lorfunc_better(fpoints, *fr[0])), '-r')
            except:
                fr = [[np.nan, np.nan, center, np.nan], []]


            w0s.append(fr[0][2])
            Qs.append(fr[0][2]/(2*fr[0][3]))

        meanw0s.append(np.mean(w0s))
        meanQs.append(np.mean(Qs))

        stdw0s.append(np.std(w0s))
        stdQs.append(np.std(Qs))

    if do_plot:
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("$|S_{21}|^2$ (dB)")
        plt.xlim(np.min(fpoints), np.max(fpoints))

    return np.array(biasV), np.array(meanw0s), np.array(meanQs), np.array(stdw0s), np.array(stdQs)


def vibrations_from_helium(dfs, reps_per_puff, fitspan=2E6, fitfunction=kfit.fitlor, fitguess=None, domains=[],
                           color='purple', puff_offsets=None, showfits=False, savename=None):
    """
    dfs : List of filenames that are loaded. Files are stitched in the order they appear in the list
    reps_per_puff : number of traces that are taken each puff
    fitspan : frequency range in Hz that is used to fit to the resonance frequency
    domains : list of start & end points that indicate a region in the figures. Ex: [[0,20], [40,60]]
    color : color scheme
    puff_offsets : list of puff offsets that are applied to the x-axis of the graphs. Must have the same length as dfs
    showfits : show the fits
    savename : filename (if figure should be saved) or None
    """
    if puff_offsets is None:
        puff_offsets = np.zeros(len(dfs))

    McRuO2 = list()
    HundredmK = list()

    meanw0s = list()
    meangammas = list()
    meanerrs = list()

    stdw0s = list()
    stdgammas = list()
    Puffs = list()

    for d, df in enumerate(dfs):
        # Get access to the data file
        if '.h5' in df:
            data = dataCacheProxy(expInst='level_meter_expt', filepath=df)

            stacks = data.index()

        idxs = list()
        for idx, s in enumerate(stacks):
            if 'stack_' in s:
                idxs.append(idx)

        stacks = np.array(stacks)[np.array(idxs)]

        Phases = np.zeros([len(stacks), 1601], dtype=np.float64)
        Mags = np.zeros([len(stacks), 1601], dtype=np.float64)
        #Puffs = np.zeros([len(stacks)], dtype=np.float64)

        plt.figure(figsize=(12., 4.))

        for idx, stack in enumerate(stacks):
            print idx,
            start_idx = (idx * reps_per_puff)
            end_idx = (idx + 1) * reps_per_puff

            data.current_stack = stack

            freq = data.get('fpoints')
            mag = data.get('mags')
            phi = data.get('phases')

            Mags[idx, :] = np.mean(mag, axis=0)
            Phases[idx, :] = np.mean(phi, axis=0)

            data.current_stack = stack
            Puffs.append(data.get('puff_nr')[0]+puff_offsets[d])
            Ts = data.get_dict('Temperatures')
            McRuO2.append(Ts['MC RuO2'])
            HundredmK.append(Ts['100mK Plate'])

            # Fitting
            w0 = list()
            gamma = list()
            errs = list()

            for j, i in enumerate(range(start_idx, end_idx)):

                try:
                    ctr = freq[j, :][np.argmax(mag[j, :])]
                    fitparams, fiterrs = fitfunction(np.array(freq[j, :], dtype=np.float64),
                                                     np.array(common.dBm_to_W(mag[j, :]), dtype=np.float64),
                                                     fitparams=fitguess, showfit=showfits,
                                                     domain=[ctr - fitspan / 2., ctr + fitspan / 2.],
                                                     mark_data='.k', verbose=False)

                    if fitfunction == kfit.fit_fano:
                        fitparams_new = [fitparams[2], fitparams[3], fitparams[0], fitparams[1]/2.]
                        fiterrs_new = [fiterrs[2], fiterrs[3], fiterrs[0], fiterrs[1]/2.]
                        fitparams = fitparams_new
                        fiterrs=fiterrs_new

                except:
                    print "fit %d failed"%idx
                    fitparams = [np.nan, np.nan, ctr, np.nan]
                    fiterrs = [np.nan, np.nan, np.nan, np.nan]

                    try:
                        ctr = freq[0, :][np.argmax(mag[j, :])]
                        fitparams, fiterrs = fitfunction(freq[0, :], common.dBm_to_W(mag[j, :]), fitparams=fitguess,
                                                          showfit=showfits, domain=[ctr - fitspan / 2., ctr + fitspan / 2.],
                                                          mark_data='.k', verbose=False)
                    except:
                        pass

                w0.append(fitparams[2])
                gamma.append(fitparams[3])
                errs.append(fiterrs)

            meanw0s.append(np.mean(w0))
            meangammas.append(np.mean(gamma))

            stdw0s.append(np.std(w0))
            stdgammas.append(np.std(gamma))

    Puffs = np.array(Puffs, dtype=np.float64)
    meanw0s = np.array(meanw0s, dtype=np.float64)
    meangammas = np.array(meangammas, dtype=np.float64)
    stdw0s = np.array(stdw0s, dtype=np.float64)
    stdgammas = np.array(stdgammas, dtype=np.float64)

    print "The final resonance frequency is %.6f GHz" % (meanw0s[-1] / 1E9)

    # Plotting
    plt.figure(figsize=(12., 4.))
    plt.subplot(121)
    common.configure_axes(13)
    plt.plot(Puffs, np.array(McRuO2) * 1E3, '-', color='purple', label='MC RuO2')
    plt.plot(Puffs, np.array(HundredmK) * 1E3, '-', color='green', label='100 mK plate')
    plt.xlabel('Puff number')
    plt.ylabel('T (mK)')
    plt.legend(loc='center left')
    plt.title('Temperature during measurements')

    plt.subplot(122)
    plt.imshow(Mags, extent=[min(freq[0, :]), max(freq[0, :]), max(Puffs), min(Puffs)], aspect='auto',
               interpolation='none', cmap=plt.cm.afmhot)
    plt.colorbar()
    plt.xlabel('Probe freq. (Hz)')
    plt.ylabel('Puff number')
    plt.title('Helium in the lines')
    plt.xlim([np.min(freq), np.max(freq)])

    fig = plt.figure(figsize=(12., 8.))
    plt.subplot(221)
    plt.errorbar(Puffs, (meanw0s) / 1E9, yerr=stdw0s / 1E9, ecolor=color, fmt='o', **common.plot_opt(color))
    ymin = np.min(meanw0s[np.logical_not(np.isnan(meanw0s))])/1E9 - 0.001
    ymax = np.max(meanw0s[np.logical_not(np.isnan(meanw0s))])/1E9 + 0.001
    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)
    plt.ylim(ymin, ymax)
    plt.xlabel('Puffs')
    plt.ylabel('$\Delta \omega_0/2\pi $ (MHz)')

    ax = plt.gca()
    ax2 = ax.twinx()
    y0 = (meanw0s[np.logical_not(np.isnan(meanw0s))][0]) / 1E9
    ax2.set_ylim((ax.get_ylim()[0]-y0)*1E3, (ax.get_ylim()[1]-y0)*1E3)
    ax2.grid()

    plt.subplot(223)
    Q = meanw0s / (2 * meangammas)
    stdQ = 1 / (2 * meangammas) * np.sqrt(stdw0s ** 2 + (meanw0s / meangammas) ** 2 * stdgammas ** 2)
    plt.errorbar(Puffs, Q, yerr=stdQ, fmt='o', ecolor=color, **common.plot_opt(color))
    for D in domains:
        plt.fill_between(D, [0, 0], y2=1.1 * np.max(Q[np.logical_not(np.isnan(Q))]), color=color, alpha=0.3)
    plt.ylim(0, 1.1 * np.max(Q[np.logical_not(np.isnan(Q))]))
    plt.xlabel('Puffs')
    plt.ylabel('Q')
    plt.grid()

    plt.subplot(222)
    plt.plot(Puffs, stdw0s / 1E3, 'o', **common.plot_opt(color))
    ymin = np.min(stdw0s[np.logical_not(np.isnan(stdw0s))])/1E3
    ymax = np.max(stdw0s[np.logical_not(np.isnan(stdw0s))])/1E3
    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)

    plt.ylim(ymin, ymax)
    plt.yscale('log')
    plt.xlabel('Puffs')
    plt.ylabel('$\sigma_{\omega_0}$ (kHz)')
    plt.grid()

    plt.subplot(224)
    plt.plot(Puffs, stdgammas / 1E3, 'o', **common.plot_opt(color))
    ymin = np.min(stdgammas[np.logical_not(np.isnan(stdgammas))])/1E3
    ymax = np.max(stdgammas[np.logical_not(np.isnan(stdgammas))])/1E3
    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)

    plt.ylim(ymin, ymax)
    plt.xlabel('Puffs')
    plt.ylabel('$\sigma_\Gamma$ (kHz)')
    plt.grid()

    if savename is not None:
        fig.savefig(savename, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(6., 4.))
    domega = meanw0s-meanw0s[np.logical_not(np.isnan(meanw0s))][0]
    plt.plot(domega/1E6, stdw0s/1E3, 'o', **common.plot_opt(color))
    plt.xlabel('$\Delta \omega/2\pi$ (MHz)')
    plt.ylabel('$\sigma_{\omega_0}$ (kHz)')
    plt.yscale('log')
    plt.grid()

    return meanw0s, meangammas, stdw0s, stdgammas, Q, stdQ


def plot_nwa_scan(df, span=1E6):
    """
    Plot a simple trace from the network analyzer and fit it with a Lorentzian
    :param df: Data filepath
    :param span: Span of the data for plotting and fitting.
    :return: f0 and Q
    """
    data = dataCacheProxy(expInst='nwa_scan', filepath=df)

    mag = data.get('mags')[0]
    freq = data.get('fpoints')[0]
    ctr = freq[np.argmax(mag)]

    fit_result, fit_err = kfit.fitlor(np.array(freq.tolist(), dtype=np.float64),
                                            common.dBm_to_W(np.array(mag.tolist(), dtype=np.float64)),
                                            showfit=True, domain=[ctr - span/2., ctr + span/2.],
                                            mark_data='.k', show_diagnostics=True)

    common.configure_axes(13)
    plt.xlim([ctr - span/2., ctr + span/2.])

    f0 = fit_result[2]
    Q = fit_result[2] / (2 * fit_result[3])
    print "f0 = %.6f GHz\nFull linewidth = %.0f kHz\nQ = %.0f" % \
          (fit_result[2] / 1E9, 2 * fit_result[3] / 1E3, fit_result[2] / (2 * fit_result[3]))

    return f0, Q


def spectrum_sweep(data_dir, fref=None, magref=None, do_plot=True, carrier=0, start_stack=None, stop_stack=None):
    """
    :param data_dir:
    :param fref:
    :param magref:
    :param do_plot:
    :param carrier: Frequency in Hz. Used to subtract from the frequency data.
    :return: Nothing
    """

    data = dataCacheProxy(expInst='spectrum_sweep', filepath=os.path.join(data_dir, 'spectrum_sweep.h5'))
    stacks = data.index()

    idxs = list()
    for idx, s in enumerate(stacks):
        if 'stack_' in s:
            stack_nr = np.int(s[6:])
            if start_stack is None and stop_stack is None:
                idxs.append(idx)
            if start_stack is None and stop_stack is not None:
                if stack_nr <= stop_stack:
                    idxs.append(idx)
            if start_stack is not None and stop_stack is None:
                if stack_nr >= start_stack:
                    idxs.append(idx)
            if start_stack is not None and stop_stack is not None:
                if stack_nr >= start_stack and stack_nr <= stop_stack:
                    idxs.append(idx)

    stacks = np.array(stacks)[np.array(idxs)]
    # print len(stacks)
    #print idxs

    Fpts = np.zeros([len(stacks), 1601], dtype=np.float64)
    Mags = np.zeros([len(stacks), 1601], dtype=np.float64)

    f = np.zeros(0)
    m = np.zeros(0)

    temperature = list()
    try:
        fdrive = data.get('fdrive')[0][np.array(idxs[1:])]
        label = 'Capacitor drive frequency (kHz)'
    except:
        fdrive = np.arange(len(stacks) - 1) * 1E3
        label = ''
        print "Did not succeed in retrieving drive frequency. You might want to specify it yourself..."

    for idx, stack in enumerate(stacks):
        try:
            #print stack
            data.current_stack = stack
            temperature.append(data.get('temperature')[0])

            if idx == 0:
                sa_cfg = data.get_dict('sa_config')
                carrier = sa_cfg['sa_center']

            freq = data.get('fpoints')[0]
            mag = data.get('mags')[0]

            Mags[idx, :] = mag
            Fpts[idx, :] = freq

            f = np.concatenate((f, Fpts[idx, :]), axis=0)
            m = np.concatenate((m, Mags[idx, :]), axis=0)
        except:
            print "Failed to get data from %s" % stack

    #print "Temperature shape"
    #print len(temperature)
    #print len(fdrive)

    if do_plot:
        fig3 = plt.figure(figsize=(10., 12.));
        common.configure_axes(13)
        plt.subplot(312)

        if fref is not None:
            #are they arrays of frequency the same?
            if np.sum(f - fref) > 0:
                print "Error, reference frequency and frequency not the same!"
            else:
                plt.plot(f - carrier, m - magref, color='b')
        else:
            plt.plot(f - carrier, m, color='b')

        fmin = np.min(Fpts[1, :] - carrier);
        fmax = np.max(Fpts[len(stacks) - 1, :] - carrier);
        plt.xlim([fmin, fmax])
        plt.xlabel('Probe frequency - carrier (Hz)')
        plt.ylabel('Power (dBm)')

        plt.subplot(311)
        plt.plot(Fpts[0, :], Mags[0, :], '-r', label='carrier')
        plt.ylabel('Power (dBm)')
        plt.legend()

        plt.subplot(313)
        plt.plot(np.array(temperature) * 1E3, 'o', **common.plot_opt('k'))
        plt.ylabel('MC temperature (mK)')
        plt.xlabel(label)

    return f, m, Fpts, Mags, fdrive


def spectrum_sweep_updated(data, fref=None, magref=None, do_plot=True, carrier=0, start_stack=None, stop_stack=None):
    """
    :param data: data_file instance
    :param fref:
    :param magref:
    :param do_plot:
    :param carrier: Frequency in Hz. Used to subtract from the frequency data.
    :return: Nothing
    """
    data.current_stack = ''

    stacks = data.index()
    total_idxs = len(stacks)
    stack_nrs = list()

    idxs = list()
    for idx, s in enumerate(stacks):
        if 'stack_' in s:
            stack_nr = np.int(s[6:])
            if start_stack is None and stop_stack is None:
                idxs.append(idx)
                stack_nrs.append(stack_nr)
            if start_stack is None and stop_stack is not None:
                if stack_nr <= stop_stack:
                    idxs.append(idx)
                    stack_nrs.append(stack_nr)
            if start_stack is not None and stop_stack is None:
                if stack_nr >= start_stack:
                    idxs.append(idx)
                    stack_nrs.append(stack_nr)
            if start_stack is not None and stop_stack is not None:
                if stack_nr >= start_stack and stack_nr <= stop_stack:
                    idxs.append(idx)
                    stack_nrs.append(stack_nr)

    stacks = np.array(stacks)[np.array(idxs)]
    print "A total of %d of %d stacks will be plotted!" % (len(stack_nrs), total_idxs - 12)

    Fpts = np.zeros([len(stacks), 1601], dtype=np.float64)
    Mags = np.zeros([len(stacks), 1601], dtype=np.float64)

    f = np.zeros(0)
    m = np.zeros(0)

    temperature = list()

    fdrive = data.get('fdrive')[0][np.array(stack_nrs)]
    carrier_f = data.get('sa_carrier_fpoints')[0]
    carrier_m = data.get('sa_carrier_mags')[0]
    label = 'Capacitor drive frequency (kHz)'

    for idx, stack in enumerate(stacks):
        try:
            data.current_stack = stack
            temperature.append(data.get('temperature')[0])

            freq = data.get('fpoints')[0]
            mag = data.get('mags')[0]

            Mags[idx, :] = mag
            Fpts[idx, :] = freq

            f = np.concatenate((f, Fpts[idx, :]), axis=0)
            m = np.concatenate((m, Mags[idx, :]), axis=0)
        except:
            print "Failed to get data from %s" % stack

    if do_plot:
        fig3 = plt.figure(figsize=(10., 12.));
        common.configure_axes(13)
        plt.subplot(312)

        if fref is not None:
            # are they arrays of frequency the same?
            if np.sum(f - fref) > 0:
                print "Error, reference frequency and frequency not the same!"
            else:
                plt.plot(f - carrier, m - magref, color='b')
        else:
            plt.plot(f - carrier, m, color='b')

        fmin = np.min(Fpts[1, :] - carrier);
        fmax = np.max(Fpts[len(stacks) - 1, :] - carrier);
        plt.xlim([fmin, fmax])
        plt.xlabel('Probe frequency - carrier (Hz)')
        plt.ylabel('Power (dBm)')

        plt.subplot(311)
        plt.plot(carrier_f, carrier_m, '-r', label='carrier')
        plt.ylabel('Power (dBm)')
        plt.xlim([np.min(carrier_f), np.max(carrier_f)])
        plt.legend()

        plt.subplot(313)
        plt.plot(np.array(temperature) * 1E3, 'o', **common.plot_opt('k'))
        plt.ylabel('MC temperature (mK)')
        plt.xlabel(label)

    return f, m, Fpts, Mags, fdrive


def integrate_peakpower(F, Mags, starts_and_stops, fdrive, includes_carrier=True, nsample=None, init_guess=None,
                        do_fit=True, sample_threshold=-90, title=""):
    """
    Integrates peak power from bin nstart to nstop. give a plot_sample number to plot one of the traces.
    This should be a bin number. fdrive is a list of drive frequencies that were applied to the capacitor.

    includes_carrier: set to True if first row of F, Mags is a spectrum that contains the carrier.
    nsample: integer that plots a sample row out of Mags and F
    """
    if nsample is not None:
        fsize = (12., 4.)
    else:
        fsize = (6., 4)

    plt.figure(figsize=fsize)
    common.configure_axes(13)

    if nsample is not None:
        plt.subplot(122)
        plt.plot(F[nsample, :], Mags[nsample, :], '-b')
        print "Trace %d passes (fdrive = %.2f kHz) the threshold of %.2f dBm at following places:" % (
            nsample, fdrive[nsample] / 1E3, sample_threshold)
        print np.where(Mags[nsample, :] > sample_threshold)[0]

    som = list()
    window_idx = 0

    for k in int(includes_carrier) + np.arange(np.shape(Mags)[0] - int(includes_carrier)):
        current_drive = fdrive[k]
        if current_drive > starts_and_stops[window_idx + 1][0]:
            window_idx += 1

        nstart = starts_and_stops[window_idx][1]
        nstop = starts_and_stops[window_idx][2]
        linear = common.dBm_to_W(Mags[k, nstart:nstop + 1])
        som.append(10 * np.log10(np.mean(linear)))

    if do_fit:
        x = fit_integrated_peakpower(fdrive, som, init_guess, color='r')

    plt.subplot(121)
    plt.plot(fdrive, som, '-k')
    plt.grid()
    plt.xlabel('Capacitor drive frequency (Hz)')
    plt.ylabel('Integrated peak power (dBm)')
    plt.title(title)

    return som


def fit_integrated_peakpower(fdrive, som, init_guess, plot_separate=True, print_fitresult=True, **kwargs):
    """

    :param fdrive: xdata
    :param som: ydata in dBm
    :param init_guess: initial guesses for fit
    :return:
    """
    fitres = kfit.fit_N_gauss(fdrive, common.dBm_to_W(np.array(som)), fitparams=init_guess)
    plt.plot(fdrive, 10 * np.log10(kfit.Ngaussfunc(fitres, fdrive)), lw=2.0, **kwargs)

    for q in range(int((len(init_guess) - 1) / 3.)):
        if plot_separate:
            plt.plot(fdrive, 10 * np.log10(kfit.gaussfunc(fitres[[0, 3 * q + 1, 3 * q + 2, 3 * q + 3]], fdrive)),
                     color='m', lw=2.0, alpha=0.3)
        if print_fitresult:
            print "--------------------------------"
            print "Q%d = %.2f" % (q + 1, fitres[3 * q + 2] / (2 * np.sqrt(2 * np.log(2)) * fitres[3 * q + 3]))
            print "f%d = %.2f kHz" % (q + 1, fitres[3 * q + 2])
            print "--------------------------------"

    return fitres


def alazar_sweep(df, min_fft_freq, max_fft_freq, ylim=None, do_imshow=False, threshold=-100):
    """
    :param data_file: dataCacheProxy instance
    :param min_fft_freq:
    :param max_fft_freq:
    :return:
    """
    data_file = dataCacheProxy(expInst='alazar_drive_sweep', filepath=df)
    data_file.current_stack = ''
    index = data_file.index()
    stacks = list()

    for I in index:
        if 'stack' in I:
            stacks.append(I)

    # Retrieve initial cavity spectrum
    fpoints = data_file.get('nwa_fpoints')[0]
    mags = data_file.get('nwa_mags')[0]
    fr = data_file.get('nwa_fit_results')[0]

    fig1 = plt.figure(figsize=(12., 8.), facecolor='white')
    plt.subplot(221)
    common.configure_axes(13)
    plt.plot(fpoints/1E9, mags, '.k')
    plt.plot(fpoints/1E9, 10 * np.log10(kfit.lorfunc(fr, fpoints)), 'r', lw=2)
    plt.xlabel('Probe freq. (Hz)')
    plt.ylabel('$|S_{21}|^2$ (dB)')
    plt.title('Cavity resonance before measurement')

    print "f0 = %.6f GHz" % (fr[2] / 1E9)

    drivepts = list()
    pump_freq = list()
    attenuation = list()
    mc_temp = list()

    print "Getting the data..."
    for idx, S in enumerate(stacks):

        pct = idx / float(len(stacks)) * 100
        if not int(pct) % 10:
            # Print the progress of loading the data
            print "%.1f%%"%pct,

        data_file.current_stack = S

        temp = data_file.get_dict('temperatures')['MC RuO2']
        attenuation.append(data_file.get('attenuation'))
        drivepts.append(data_file.get('drive_frequency'))
        pump_freq.append(data_file.get('pump_frequency'))
        mc_temp.append(temp)

        FFT = data_file.get('averaged_fft')[0]
        freq = data_file.get('fft_points')[0]

        if idx == 0:
            print "Number of FFT points: %d" % (len(freq))
            min_fft_idx = common.find_nearest(freq, min_fft_freq)
            max_fft_idx = common.find_nearest(freq, max_fft_freq)

            print "Only taking data from idx %d to %d" % (min_fft_idx, max_fft_idx)

            fft_freq = freq[min_fft_idx:max_fft_idx]
            fft = np.zeros([0, len(fft_freq)])

        fft = np.vstack((fft, np.abs(FFT[min_fft_idx:max_fft_idx]) ** 2))

    drivepts = np.array(drivepts)
    pump_freq = np.array(pump_freq)
    attenuation = np.array(attenuation)

    # To convert from dBm to dBm/Hz.
    data_file.current_stack = ''
    alazar_dict = data_file.get_dict('alazar_cfg')
    T = alazar_dict['samplesPerRecord']*1/(alazar_dict['sample_rate']*1E3)
    fft = fft*T

    print np.shape(mc_temp)
    ylims = [plt.ylim()[0], plt.ylim()[1]]
    plt.fill_between([(fr[2] - 2 * drivepts[0][0])/1E9, (fr[2] - 2 * drivepts[-1][0])/1E9],
                     [ylims[0], ylims[0]], y2=[ylims[1], ylims[1]], color='red',
                     alpha=0.2)
    plt.ylim(ylims)
    plt.xlim(np.min(fpoints)/1E9, np.max(fpoints)/1E9)

    plt.subplot(222)
    plt.plot(drivepts / 1E3, (pump_freq - fr[2]) / 1E3, 'o', **common.plot_opt('orange'))
    plt.xlabel('Drive frequency (kHz)')
    plt.ylabel('Pump detuning from resonance$\Delta$ (kHz)')

    plt.subplot(223)
    plt.plot(drivepts / 1E3, attenuation, 'o', **common.plot_opt('orange'))
    plt.xlabel('Drive frequency (kHz)')
    plt.ylabel('Attenuation (dB)')

    plt.subplot(224)
    plt.plot(drivepts / 1E3, np.array(mc_temp) * 1E3, 'o', **common.plot_opt('orange'))
    plt.xlabel('Drive frequency (kHz)')
    plt.ylabel('MC temperature (mK)')
    fig1.savefig(os.path.join(os.path.split(df)[0], 'fig1.png'), dpi=200)

    return fft_freq, fft, drivepts


def plot_alazar_sweep(fft_freq, fft, drivepts, min_fft_freq, savepath=None, threshold=-100, do_imshow=True,
                      ylim=None, xlim=None, do_colorbar=False, do_dumb=True, do_plot_2omega=False, do_plot_1omega=False):
    """
    fft_freq:       1D array with FFT frequencies, obtained from alazar_sweep.
    fft:            2D array containing FFT**2, from alazar_sweep
    drivepts:       Drive frequencies of the source
    min_fft_freq:   Minimum FFT frequency you would like to display in your plots in Hz.
    savepath:       Datapath, where to save the figures. Default: None
    threshold:      Determines the threshold in dBm/Hz for traces to be marked as interesting. Interesting traces are
                    plot separately.
    do_imshow:      Plot the 2d array "fft"
    ylim:           List of color limits for the colorplot and for the interesting traces. Ex: [-120, 90]. Default: None
    do_colorbar:    Plot a colorbar for the 2d colorplot. Takes up a little more space, so can be disabled by setting
                    do_colorbar=False.
    """
    selected_fft = 10 * np.log10(fft)[:,np.logical_and(fft_freq>xlim[0]*1E3, fft_freq<xlim[1]*1E3)]
    #print np.shape(selected_fft)

    print "Delta f = {} Hz".format(fft_freq[1]-fft_freq[0])

    if do_imshow:
        fig2 = plt.figure(figsize=(16., 8.), facecolor='white')
        common.configure_axes(13)
        plt.imshow(selected_fft, aspect='auto', interpolation='none',
                   extent=[xlim[0], xlim[1], drivepts[-1]/1E3, drivepts[0]/1E3],
                   vmin=ylim[0], vmax=ylim[1])
        if do_colorbar:
            plt.colorbar()
        plt.title('Power spectral density (dBm/Hz)')
        plt.xlabel('FFT frequency (kHz)')
        plt.ylabel('Drive frequency (kHz)')

        if ylim is not None and do_colorbar:
            plt.clim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

        if do_plot_2omega:
            plt.plot(2*drivepts/1E3, drivepts/1E3, '-', color='white', alpha=0.25, lw = 5)
        if do_plot_1omega:
            plt.plot(drivepts/1E3, drivepts/1E3, '-', color='white', alpha=0.25, lw = 5)

        if savepath is not None:
            fig2.savefig(os.path.join(savepath, 'fig2.png'), dpi=200)
        #plt.xlim([fft_freq[0], fft_freq[-1]]);

    above_noise = np.where(selected_fft > threshold)
    # A trace is marked interesting when there's a a frequency bin with value > threshold dB
    # That frequency may not be the 0 Hz
    if min_fft_freq == 0:
        interesting_traces = np.unique(above_noise[0][np.where(above_noise[1] > 0)])
    else:
        interesting_traces = np.unique(above_noise[0])

    from mpltools import color

    if do_dumb:
        fig4 = plt.figure(figsize=(8.,6.), facecolor='white')
        color.cycle_cmap(length=np.shape(selected_fft)[0], cmap=plt.cm.jet)
        ax = plt.gca()
        for idx, i in enumerate(range(np.shape(selected_fft)[0])):
            ax.plot(fft_freq[np.logical_and(fft_freq>xlim[0]*1E3, fft_freq<xlim[1]*1E3)] / 1E3, selected_fft[i,:])
            plt.ylim(ylim)
            plt.xlim(xlim)
        plt.xlabel('FFT frequency (kHz)')
        plt.ylabel('PSD (dBm/Hz)')

    nrows = np.ceil(len(interesting_traces)/4.)
    fig3 = plt.figure(figsize=(16, 2*nrows), facecolor='white')
    for idx,INT in enumerate(interesting_traces):
        plt.subplot(nrows, 4, idx)
        plt.plot(fft_freq[np.logical_and(fft_freq>xlim[0]*1E3, fft_freq<xlim[1]*1E3)] / 1E3, selected_fft[INT, :])
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.annotate('%.0f kHz'%(drivepts[INT]/1E3) , xy=(1, 1), xycoords='axes fraction', fontsize=16,
                horizontalalignment='right', verticalalignment='top', color='r')

    #plt.xlabel('FFT frequency (kHz)')
    #plt.ylabel('PSD (dBm/Hz)')

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    if savepath is not None:
        fig3.savefig(os.path.join(savepath, 'fig3.png'), dpi=200)

    print "There are %d interesting traces" % len(interesting_traces)

    return interesting_traces


def study_vibrations(d, showfit=False, do_plot=True, verbose=True):
    """
    d       : filepath of the file, including filename.
    showfit : default = False. Shows all the fits.
    do_plot : makes a plot of all the obtained fit values.
    """

    data = dataCacheProxy(expInst='studying_vibrations', filepath=d)
    temps = data.get_dict('Temperatures')

    if verbose:
        print "Temperatures are as follows:\n4K plate: \t {} K\nStill: \t\t {} K\n100 mK plate: \t {} mK\nBase: \t\t {} mK"\
        .format(temps['PT2 Plate'], temps['Still'], temps['100mK Plate']*1E3, temps['MC RuO2']*1E3)

    fpoints = data.get('fpoints')
    mags = data.get('mags')
    w0 = list()
    Q = list()

    if showfit:
        plt.figure(figsize=(14.,6))
        common.configure_axes(13)

    for k in range(np.shape(mags)[0]):
        f = fpoints[0,:]
        m = mags[k,:]

        center=f[np.argmax(m)]
        span = 1.0E6

        fitres, fiterr = kfit.fitlor(f, common.dBm_to_W(m), showfit=showfit, domain=[center-span/2., center+span/2.], verbose=False)

        w0.append(fitres[2])
        Q.append(fitres[2]/(2*fitres[3]))

    if do_plot:
        plt.figure()
        plt.plot(np.array(w0)/1E9, 'sk')
        plt.fill_between([0, np.shape(mags)[0]], [np.mean(w0)/1E9 - np.std(w0)/1E9, np.mean(w0)/1E9 - np.std(w0)/1E9],
                         y2 = [np.mean(w0)/1E9 + np.std(w0)/1E9, np.mean(w0)/1E9 + np.std(w0)/1E9],
                         color='lightgreen', alpha=0.3, lw=0)
        plt.ylabel('$\omega_0/2\pi$ (GHz)')

    print "Mean is {} GHz, standard deviation is {} kHz".format(np.mean(w0)/1E9, np.std(w0)/1E3)

    return np.mean(w0), np.std(w0), np.mean(Q), np.std(Q), temps


def plot_spectrum(d, freqlim, channel='ch1', verbose=True):
    """
    Plot a spectrum from a datafile with file path "d"
    Channel may be 'ch1' or 'ch2'
    """
    data = dataCacheProxy(expInst='', filepath=d)
    t = data.get('t')[0]
    ch1 = data.get(channel)[0]

    freq, Y = common.plot_spectrum(ch1, t, freqlim=freqlim, verbose=verbose)
    plt.grid()

    if verbose:
        print "Delta f = {} Hz".format(freq[1]-freq[0])

    return freq, Y


def anal_cavityS21_expt(df, ch1range, ch2range, verbose=True, do_log=False, ylim='auto', xlim='auto', multiply=1):

    data = dataCacheProxy(expInst='cavity_S21_double', filepath=df)
    t = data.get('t')
    ch1 = data.get('ch1')
    ch2 = data.get('ch2')

    Ych1 = np.zeros((np.shape(ch1)[0], np.shape(ch1)[1]/2))
    Ych2 = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1]/2))

    for stack in range(np.shape(ch1)[0]):
        print "stack %d/%d" % (stack+1, np.shape(ch1)[0])
        f, Y = common.plot_spectrum(ch1[stack,:], t[stack,:], ret=True, do_plot=False, verbose=verbose)
        Ych1[stack,:] = np.abs(Y) * np.sqrt(np.max(t[stack,:])) * multiply

        f, Y = common.plot_spectrum(ch2[stack,:], t[stack,:], ret=True, do_plot=False, verbose=verbose)
        Ych2[stack,:] = np.abs(Y) * np.sqrt(np.max(t[stack,:]))

    meanYch1 = np.mean(Ych1, axis=0)
    meanYch2 = np.mean(Ych2, axis=0)

    plt.figure(figsize=(6.,4.))
    plt.plot(f, meanYch1, '-r', label='Ch1')
    plt.plot(f, meanYch2, '-b', label='Ch2')
    if xlim != 'auto':
        plt.xlim(xlim)

    if ylim != 'auto':
        plt.ylim(ylim)

    if do_log:
        plt.yscale('log')

    plt.xlabel('FFT freq (Hz)')
    plt.ylabel(r'Amplitude spectral density (V/$\sqrt{\mathrm{Hz}}$)')

    return t, ch1, ch2, f, meanYch1, meanYch2


def process_level_meter_s11(dfs, reps_per_puff, fitspan=2E6, fitmode='twoport', fitguess=None,
                            puff_offsets=None, showfits=False):
    """
    Process the level meter data measured in a Hybrid setup, measuring reflection from the caviy.
    :param dfs: List of data files containing level meter data
    :param reps_per_puff: Repetitions per helium puff
    :param fitspan: Fit span in Hz
    :param fitmode: 'twoport' or 'oneport'
    :param fitguess: Default is None, If supplied has to be a list [f0, Qc, Qi, df, scale]
    :param puff_offsets: List
    :param showfits: True/False
    :return: Puffs, mean_fitparams, err_fitparams
    """
    if puff_offsets is None:
        puff_offsets = np.zeros(len(dfs))

    McRuO2 = list()
    HundredmK = list()

    mean_fitparams = {"f0" : list(), "Qc" : list(), "Qi" : list(), "df" : list()}
    err_fitparams = {"f0" : list(), "Qc" : list(), "Qi" : list(), "df" : list()}

    Puffs = list()

    for d, Df in enumerate(dfs):
        # Get access to the data file
        if '.h5' in Df:
            data = dataCacheProxy(expInst='level_meter_expt', filepath=Df)

            stacks = data.index()

        idxs = list()
        for idx, s in enumerate(stacks):
            if 'stack_' in s:
                idxs.append(idx)

        stacks = np.array(stacks)[np.array(idxs)]

        Phases = np.zeros([len(stacks), 1601], dtype=np.float64)
        Mags = np.zeros([len(stacks), 1601], dtype=np.float64)
        Fpoints = np.zeros([len(stacks), 1601], dtype=np.float64)

        plt.figure(figsize=(12., 4.))

        for idx, stack in enumerate(stacks):
            print idx,
            start_idx = (idx * reps_per_puff)
            end_idx = (idx + 1) * reps_per_puff

            data.current_stack = stack

            freq = data.get('fpoints')
            mag = data.get('mags')
            phi = data.get('phases')

            Mags[idx, :] = np.mean(mag, axis=0)
            Phases[idx, :] = np.mean(phi, axis=0)
            Fpoints[idx, :] = freq[0,:]

            data.current_stack = stack
            Puffs.append(data.get('puff_nr')[0]+puff_offsets[d])
            Ts = data.get_dict('Temperatures')
            McRuO2.append(Ts['MC RuO2'])
            HundredmK.append(Ts['100mK Plate'])

            # Fitting
            f0 = list(); Qc = list(); Qi = list(); df = list();

            for j, i in enumerate(range(start_idx, end_idx)):

                ctr = freq[j, :][np.argmin(mag[j, :])]
                # This part is still quite specific to fit_s11 from kfit
                try:
                    normalized_mags = common.dBm_to_W(mag[j, :]-np.mean(mag[j,:-100]))*1E3
                    fitparams, fiterrs = kfit.fit_s11(np.array(freq[j, :], dtype=np.float64),
                                                       np.sqrt(normalized_mags), mode=fitmode,
                                                       fitparams=fitguess, showfit=showfits,
                                                       domain=[ctr - fitspan / 2., ctr + fitspan / 2.],
                                                       mark_data='.k', verbose=False)

                    if fitmode == "twoport":
                        # fitparams = [f0, Qc, Qi, df, scale]
                        f0.append(fitparams[0])
                        Qc.append(fitparams[1])
                        Qi.append(fitparams[2])
                        df.append(fitparams[3])
                    elif fitmode == "oneport":
                        # fitparams = [f0, kr, eps, df, scale]
                        f0.append(fitparams[0])
                        Qc.append(fitparams[0]/(fitparams[1]))
                        Qi.append(fitparams[0]/(2*fitparams[2]))
                        df.append(fitparams[3])

                except:
                    pass

            mean_fitparams['f0'].append(np.mean(f0))
            mean_fitparams['Qc'].append(np.mean(Qc))
            mean_fitparams['Qi'].append(np.mean(Qi))
            mean_fitparams['df'].append(np.mean(df))

            err_fitparams['f0'].append(np.std(f0))
            err_fitparams['Qc'].append(np.std(Qc))
            err_fitparams['Qi'].append(np.std(Qi))
            err_fitparams['df'].append(np.std(df))

    # Make sure they're numpy arrays instead of lists
    Puffs = np.array(Puffs, dtype=np.float64)

    for key in mean_fitparams.keys():
        mean_fitparams[key] = np.array(mean_fitparams[key], dtype=np.float64)
        err_fitparams[key] = np.array(err_fitparams[key], dtype=np.float64)


    plt.figure(figsize=(12., 4.))
    plt.subplot(121)
    common.configure_axes(13)
    plt.plot(Puffs, np.array(McRuO2) * 1E3, '-', color='purple', label='MC RuO2')
    plt.plot(Puffs, np.array(HundredmK) * 1E3, '-', color='green', label='100 mK plate')
    plt.xlabel('Puff number')
    plt.ylabel('T (mK)')
    plt.legend(loc='center left')
    plt.title('Temperature during measurements')

    if len(dfs) == 1:
        plt.subplot(122)
        plt.pcolormesh(Puffs, np.transpose(Fpoints), np.transpose(Mags), cmap=plt.cm.viridis)
        plt.colorbar()
        plt.ylabel('Probe freq. (Hz)')
        plt.xlim(min(Puffs), max(Puffs))
        plt.ylim(np.min(Fpoints), np.max(Fpoints))
        plt.xlabel('Puff number')
        plt.title('Helium in the lines')

    print "\nThe final resonance frequency is %.6f GHz" % (mean_fitparams['f0'][-1] / 1E9)

    return Puffs, mean_fitparams, err_fitparams


def analyze_level_meter_s11(Puffs, mean_fitparams, err_fitparams, color='deeppink', domains=[], save_path=None):
    """
    Next step after processing the level meter curve, from process_level_meter_s11
    :param Puffs: Array containing the puff numbers
    :param mean_fitparams: Mean fit parameters from process_level_meter_s11
    :param err_fitparams: Standard deviation from fit parameters from process_level_meter_s11
    :param color: string, color to use
    :param domains: List of tuples indicating different fill domains
    :param save_path: File name for saving. Default is don't save the figure.
    :return: None.
    """
    f0 = mean_fitparams['f0']

    fig1 = plt.figure(figsize=(12., 8.))
    plt.subplot(221)
    plt.errorbar(Puffs, f0/1E9, yerr=err_fitparams['f0']/1E9, ecolor=color, fmt='o', **common.plot_opt(color))
    try:
        ymin = np.min(f0[np.logical_not(np.isnan(f0))])/1E9 - 0.001
        ymax = np.max(f0[np.logical_not(np.isnan(f0))])/1E9 + 0.001
    except:
        print "meanw0s has only NaN entries. Please check your data."

    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)
    plt.ylim(ymin, ymax)
    plt.xlabel('Puffs')
    plt.ylabel(r'$ \omega_0/2\pi $ (GHz)')

    ax = plt.gca()
    ax2 = ax.twinx()
    y0 = (f0[np.logical_not(np.isnan(f0))][0]) / 1E9
    ax2.set_ylim((ax.get_ylim()[0]-y0)*1E3, (ax.get_ylim()[1]-y0)*1E3)
    ax2.set_ylabel(r'$\Delta \omega_0/2\pi $ (MHz)')
    ax2.grid()

    plt.subplot(223)
    Qc = mean_fitparams['Qc']
    plt.errorbar(Puffs, Qc, yerr=err_fitparams['Qc'], fmt='o', ecolor=color, **common.plot_opt(color))
    for D in domains:
        plt.fill_between(D, [0, 0], y2=1.1 * np.max(Qc[np.logical_not(np.isnan(Qc))]), color=color, alpha=0.3)
    plt.ylim(0, 1.1 * np.max(Qc[np.logical_not(np.isnan(Qc))]))
    plt.xlabel('Puffs')
    plt.ylabel('$Q_c$')
    plt.grid()

    plt.subplot(222)
    stdf0 = err_fitparams['f0']
    plt.plot(Puffs, stdf0/1E3, 'o', **common.plot_opt(color))
    ymin = np.min(stdf0[np.logical_not(np.isnan(stdf0))])/1E3
    ymax = np.max(stdf0[np.logical_not(np.isnan(stdf0))])/1E3
    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.ylim(ymin, ymax)
    plt.yscale('log')
    plt.xlabel('Puffs')
    plt.ylabel('$\sigma_{\omega_0}$ (kHz)')
    plt.grid()

    plt.subplot(224)
    Qi = mean_fitparams['Qi']
    plt.plot(Puffs, Qi, 'o', **common.plot_opt(color))
    try:
        ymin = np.min(Qi[np.logical_not(np.isnan(Qi))])
        ymax = np.max(Qi[np.logical_not(np.isnan(Qi))])
    except:
        print "meanw0s has only NaN entries. Please check your data."
    for D in domains:
        plt.fill_between(D, [ymin, ymin], y2=ymax, color=color, alpha=0.3)

    plt.ylim(ymin, ymax)
    plt.xlabel('Puffs')
    plt.ylabel('$Q_i$')
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


    if save_path is not None:
        common.save_figure(fig1, save_path=save_path)

    fig2=plt.figure(figsize=(6., 4.))
    domega = f0-f0[np.logical_not(np.isnan(f0))][0]
    plt.plot(domega/1E6, err_fitparams['f0']/1E3, 'o', **common.plot_opt(color))
    plt.xlabel('$\Delta \omega/2\pi$ (MHz)')
    plt.ylabel('$\sigma_{\omega_0}$ (kHz)')
    plt.yscale('log')
    plt.grid()

    if save_path is not None:
        common.save_figure(fig2, save_path=save_path)


if __name__ == '__main__':
    pass
