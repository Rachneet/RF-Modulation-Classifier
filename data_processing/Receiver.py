#!/usr/bin/python
# -*- coding: utf-8 -*-

import abc
from data_processing.Radio import Radio
from data_processing.helper_functions import info, debug, warning, error, critical
from visualization.colors import color_array,color_bw_noise,color_bw_signal

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import numpy.fft as fft
from matplotlib.ticker import ScalarFormatter


class Receiver(Radio):
    """ Generic receiver class """

    def __init__(self,
                 freq_noise,
                 bw_noise,
                 srate,
                 rx_dir,
                 **kwargs):

        self.data = None

        # save positional arguments
        self.freq_noise = freq_noise
        self.bw_noise = bw_noise
        self.srate = srate
        self.rx_dir = rx_dir
        # call the constructor of the parent
        Radio.__init__(self, **kwargs)

        # member variables
        self.freq_noise_offset = freq_noise - self.freq
        self.signal_power = None
        self.noise_power = None
        self.interferer_power = None

    @abc.abstractmethod
    def receive_samples(self, n_samples, filename, verbose=True):
        """Receives a number of samples from the hardware
        and saves them in self.data"""

    def calculate_spectrum(self, n_fft=10000, window='hann', data=None):
        """Calculates the spectrum based on sampled IQ data. Spectra are
        averaged in the power domain (RMS) to get a low variance estimate
        The implementation follows: https://holometer.fnal.gov/GH_FFT.pdf
        """

        # check if data is valid
        if(data is None):
            data = np.copy(self.data)
        else:
            data = np.copy(data)
        # print(data)
        # reshape samples for averaging
        if(n_fft <= data.size):
            data = np.reshape(data, (-1, n_fft))
            n_window = n_fft
        else:
            n_window = data.size

        # window function
        w = scipy.signal.get_window(window, n_window)
        s1 = np.sum(w)
        s2 = np.sum(w**2)
        enbw = self.srate * (s2 / s1**2)
        data *= w

        # calculate spectrum by fft
        data_f = fft.fftshift(fft.fft(data, n=n_fft))
        f = fft.fftshift(fft.fftfreq(n=n_fft, d=1 / self.srate))
        spectrum = abs(data_f)**2

        # average in power domain
        if(n_fft <= data.size):
            spectrum = np.average(spectrum, axis=0)

        # normalizations
        spectrum /= 2  # rms
        spectrum /= 50  # 50 Ohm
        spectrum /= 0.001  # 1 mW
        spectrum /= s1**2  # window

        # interpolate DC carrier
        # idx = np.argmin(spectrum)
        # spectrum[idx] = 0.5*(spectrum[idx-1]+spectrum[idx+2])

        return (f, spectrum)

    def receive_and_filter(self, n_samples):

        self.receive_samples(n_samples, "cache.bin", verbose=False)
        return self.filter_data()

    def filter_data(self, data=None, plot=False):
        """low-pass filters the data within the signal bandwidth"""

        # check if data is valid
        if(data is None):
            data = self.data

        # filter configuration
        numtaps = 100
        b = scipy.signal.firwin(numtaps=numtaps,
                                cutoff=self.bw_signal / 2,
                                pass_zero=True,
                                fs=self.srate,
                                )
        self.b = b
        # show frequency response of the filter
        if(plot):

            w, h = scipy.signal.freqz(b, worN=1500)
            w *= self.srate / (2 * np.pi) / 1e6  # w in MHz

            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 7))

            for ax, y, label in zip([ax1, ax2],
                             [20 * np.log10(abs(h)), np.unwrap(np.angle(h)) / np.pi],
                             ["Magnitude [dB]", "Phase [$\\pi^{-1}$]"]):

                ax.plot(w, y)
                ax.set_xlim([0, min(self.srate / 2e6, (self.freq_noise[1] + 2 * self.bw_noise) / 1e6)])
                ax.fill_between([0, self.bw_signal / 1e6 / 2], -200, 200,
                                facecolor=color_bw_signal, label='Signal band')
                noise_offset = self.freq_noise[1] - self.freq
                noise_freq_idx = np.array([(noise_offset - self.bw_noise / 2) / 1e6,
                                           (noise_offset + self.bw_noise / 2) / 1e6])
                ax.fill_between(noise_freq_idx, -200, 200,
                                facecolor=color_bw_noise, label='Noise band')
                ax.set_ylim([min(y), max(y) + 0.1 * abs(min(y))])
                ax.set_ylabel(label)
                ax.set_title("Frequency response of the low-pass filter")
                ax.set_xlabel("Frequency [MHz]")
                ax.legend(loc=1)

            ax1.set_ylim([-80, 5])

            # -3dB point
            x, y = w[np.argmin(abs(abs(h) - np.sqrt(0.5)))], 20 * np.log10(np.sqrt(0.5))
            ax1.plot(x, y, 'ob', markersize=4)
            ax1.annotate('%5.2fdB @\n%5.2f MHz' % (y, x),
                xy=(x, y),
                xytext=(10, 0),
                textcoords='offset points',
                ha='left',
                va='center')

            # attenuation @ f_c
            x, y = self.bw_signal / 2e6, 20 * np.log10(abs(h[np.argmin(abs(w - self.bw_signal / 2e6))]))
            ax1.plot(x, y, 'og', markersize=4)
            ax1.annotate('%5.2fdB @\n%5.2f MHz' % (y, x),
                xy=(x, y),
                xytext=(-10, -5),
                textcoords='offset points',
                ha='right',
                va='center')

            # average noise attenuation
            noise_idx = np.array(np.arange(noise_freq_idx[0] * (w.size / (self.srate / 2e6)),
                                           noise_freq_idx[1] * (w.size / (self.srate / 2e6)), 1), dtype=int)
            avg_mag_noise = 10 * np.log10(np.mean(abs(h[noise_idx])**2))
            ax1.plot(noise_freq_idx, [avg_mag_noise] * 2, 'r')

            x, y = np.mean(noise_freq_idx), avg_mag_noise
            ax1.annotate('average noise\nattenuation: %5.2fdB' % avg_mag_noise,
                xy=(x, y),
                xytext=(25, 30),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8, headlength=8),
                textcoords='offset points',
                ha='center',
                va='bottom')

            plt.tight_layout()
            plt.show()

        self.filtered_data = scipy.signal.convolve(data, b, mode='same')

        return self.filtered_data

    def visualize_samples(self, mode='unfiltered', n_time=None, n_fft=1024,  # I changed this
                          plot=True, data=None, filtered_data=None, window='hann'):
        """visualizes the data in time and frequency domain
        n_time: number of samples to plot in time domain
        n_fft: number of samples for fft
        mode: {'filtered', 'unfiltered', 'both'}"""

        # booleans / mode
        FILTERED = (mode == 'filtered')
        UNFILTERED = (mode == 'unfiltered')
        BOTH = (mode == 'both')
        mode = 'Unfiltered' if UNFILTERED else 'Filtered'

        # check if data is valid
        if(n_time is None):
            n_time = int(20 * self.srate / 10e6)  # show 10 symbols

        if((UNFILTERED or BOTH) and data is None):
            data = self.data
            assert n_time <= data.size, "Trying to plot more data than there is"

        if((FILTERED or BOTH) and filtered_data is None):
            filtered_data = self.filtered_data
            assert filtered_data.size == data.size, "data and filtered_data have to have the same length"

        # color helper
        full_tone = (c for c, i in zip(color_array, range(len(color_array))) if i % 2 == 0)
        shaded = (c for c, i in zip(color_array, range(len(color_array))) if i % 2 == 1)

        # plots
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        stime = 1 / self.srate
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        x = np.arange(0, n_time) * stime

        # plot spectrum
        bw_signal_lim = [-self.bw_signal / (2 * 1e6), self.bw_signal / (2 * 1e6)]
        bw_noise_lim_low = [(-(self.bw_noise / 2) + self.freq_noise_offset[0]) / 1e6,
                            ((self.bw_noise / 2) + self.freq_noise_offset[0]) / 1e6]
        bw_noise_lim_high = [(-(self.bw_noise / 2) + self.freq_noise_offset[1]) / 1e6,
                            ((self.bw_noise / 2) + self.freq_noise_offset[1]) / 1e6]


        if(UNFILTERED or BOTH):
            color = next(shaded) if BOTH else next(full_tone)
            label = None if BOTH else "Power spectrum"
            fp, per = self.calculate_spectrum(n_fft=n_fft, data=data, window=window)
            ax2.plot(fp / 1e6, 10 * np.log10(per), label=label, color=color)
        if(FILTERED or BOTH):
            # print(filtered_data)
            fp, per = self.calculate_spectrum(n_fft=n_fft, data=filtered_data, window=window)
            ax2.plot(fp / 1e6, 10 * np.log10(per), label='Power spectrum', color=next(full_tone))

        ax2.fill_betweenx([-250, 0], *bw_signal_lim, facecolor=color_bw_signal, label='Signal band')
        ax2.fill_betweenx([-250, 0], *bw_noise_lim_low, facecolor=color_bw_noise, label='Noise band')
        ax2.fill_betweenx([-250, 0], *bw_noise_lim_high, facecolor=color_bw_noise)
        ax2.set_ylim([min(10 * np.log10(per)), 0.95 * max(10 * np.log10(per))])
        ax2.set_xlim(np.array([-1, 1]) * min(4 * self.bw_signal / (2e6), self.srate / 2e6))
        ax2.set_xlabel("Frequency [MHz]")
        ax2.set_ylabel("Spectrum [$dBm_{50Ω}$] per bin")
        ax2.set_title("%s spectrum (Baseband)" % mode)
        ax2.legend(loc=2)

        # plot time domain

        if(UNFILTERED or BOTH):
            color = shaded if BOTH else full_tone
            label = iter([None, None]) if BOTH else iter(['I', 'Q'])
            ax1.plot(x, data[:+n_time].real, label=next(label), color=next(color))
            ax1.plot(x, data[:+n_time].imag, label=next(label), color=next(color))
        if(FILTERED or BOTH):
            ax1.plot(x, filtered_data[:n_time].real, label='I', color=next(full_tone))
            ax1.plot(x, filtered_data[:n_time].imag, label='Q', color=next(full_tone))

        ax1.set_xlim([0, stime * (n_time - 1)])
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [V]")
        ax1.set_title("%s time Domain" % mode)
        ax1.legend(loc=2)
        ax1.xaxis.set_major_formatter(xfmt)

        if(plot):
            plt.show()

        return (ax1, ax2)

    def visualize_BPSK_fullframe(self, data=None):
        """visualizes the flatness of a full BPSK sample run"""

        if (data is None):
            if (self.filtered_data is None):
                error("self.name", "BPSK visualization failed - no data")
            else:
                data = self.filtered_data

        # color helper
        color = (c for c, i in zip(color_array, range(len(color_array))) if i % 2 == 0)

        # plots
        n_time = data.size
        stime = 1 / self.srate
        N = int(n_time / 100)
        fig, ax = plt.subplots(figsize=(16, 2))
        x = np.arange(0, n_time) * stime * 1e3
        plotdata = data**2  # remove information
        plotdata = scipy.signal.convolve(plotdata, np.ones((N,)) / N, mode='same')  # lowpass filter

        # plot time domain
        ax.plot(x[::10], plotdata.real[::10], label='I', color=next(color))
        ax.plot(x[::10], plotdata.imag[::10], label='Q', color=next(color))
        ax.plot(x[::10], abs(plotdata)[::10], label='abs', color=next(color))

        ax.set_xlim([min(x), max(x)])
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Voltage² [V²]")
        ax.set_title("BPSK Envolope in Time Domain")
        ax.legend(loc=2)

        plt.show()

    def measure_powers(self, n_fft=10000, window='hann'):
        """
        Calculates band and noise power based on
        the sampled IQ data. The noise power density is measured within
        a given BW, with certain offset from the carrier.

        returns: (band_power [dBm], noise_power [dBm], noiseless_band_power [dBm])
        """

        # return values
        band_power = None
        noise_power = None

        # check if data is valid
        assert self.data is not None, "Data not set"
        assert n_fft <= self.data.size, "Trying to fft more data than there is"

        _, per = self.calculate_spectrum(n_fft=n_fft, data=self.data, window=window)

        # extract band power (bw_signal)
        mid = int(len(per) / 2)  # DC carrier
        step = self.srate / n_fft
        steps = self.bw_signal / step
        idx_band = np.array(np.arange(steps) + mid - steps / 2, dtype=int)
        band_power = 10 * np.log10(np.sum(per[idx_band]))

        # extract noise power
        # the noise power is measure in bw_noise Hertz at freq_noise Hz
        # and then multiplied by bw_signal/bw_noise to get the approximate
        # noise power in the signal bandwidth
        step = self.srate / n_fft
        steps = self.bw_noise / step
        offset_steps_low = (self.freq_noise[0] - self.freq) / step
        offset_steps_high = (self.freq_noise[1] - self.freq) / step
        idx_noise_low = np.array(np.arange(steps) + mid - steps / 2 + offset_steps_low, dtype=int)
        idx_noise_high = np.array(np.arange(steps) + mid - steps / 2 + offset_steps_high, dtype=int)
        bw_factor = self.bw_signal / (2*self.bw_noise)  # factor between signal and noise measurement BW

        noise_power = 10 * np.log10(bw_factor * (np.sum(per[idx_noise_low]) + np.sum(per[idx_noise_high])))

        if(band_power < noise_power):
            error(self.name, "band_power: %5.2f dB\tnoise_power: %5.2f dB" % (band_power, noise_power))
            error("sys", "Illegal values in measure_powers()!")
            raise Exception("Illegal power values")

        # calculate the noiseless_band_power
        # this is the pure signal / interferer power based on the transmitter
        noiseless_band_power = 10 * np.log10(10**(band_power / 10) -
                                           10**(noise_power / 10))

        debug(self.name, "band_power: %5.2f dB\tnoise_power: %5.2f dB\tsignal_power: %5.2f dB\t"
            % (band_power, noise_power, noiseless_band_power))

        return (band_power, noise_power, noiseless_band_power)

    def measure_SNR(self):
        """measures noise power, signal power and returns snr"""

        _, self.noise_power, self.signal_power = self.measure_powers()
        self.snr = self.signal_power - self.noise_power
        debug(self.name, "SNR: %5.3f dB" % self.snr)
        return self.snr

    def measure_SIR(self):
        """measures noise power, interferer power and returns sir
        Assumes a valid and up-to-date value in self.signal_power
        (run measure_SNR first)"""

        assert self.signal_power is not None, \
            "SIR calculation depends on self.signal_power. " + \
            "Did you run measure_SNR first?"

        _, _, self.interference_power = self.measure_powers()
        self.sir = self.signal_power - self.interference_power
        debug(self.name, "SIR: %5.3f dB" % self.sir)
        return self.sir

    def measure_SINR(self):
        """returns the receiver SINR
        Assumes a valid and up-to-date value in self.signal_power &
        self.interferer_power. (run measure_SNR / measure_SIR first)"""

        assert self.signal_power is not None, \
            "SIR calculation depends on self.signal_power. " + \
            "Did you run measure_SNR first?"

        assert self.interference_power is not None, \
            "SIR calculation depends on self.interference_power. " + \
            "Did you run measure_SIR first?"

        interf_noise_power = 10 * np.log10(10**(self.interference_power / 10) +
                                           10**(self.noise_power / 10))
        self.sinr = self.signal_power - interf_noise_power
        debug(self.name, "SINR: %5.3f dB" % self.sinr)
        return self.sinr

    def get_cfo(self, search_bw, plot=False):
        """
        Calculates the CFO between receiver and transmitter
        It is assumed that the transmitter is transmitting a single tone at freq_tx
        The receiver then records a band around that frequency and
        report the relative frequency of the maximum fft peak
        """

        try:
            self.backup_cfo = self.get_attributes()  # back up all attributes
            self.srate = search_bw
            self.receive_samples(int(10 * search_bw), "cache.bin", verbose=False)
            fp, per = self.calculate_spectrum(int(search_bw), window='boxcar')
            cfo = fp[np.argmax(per)]
        except Exception as err:
            raise err
        finally:
            self.update(**self.backup_cfo)

        debug(self.name, "Measured CFO: %d Hz" % int(cfo))

        if(plot):
            # plot spectrum
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(fp, 10 * np.log10(per), label='Power spectrum')
            ax.set_ylim([min(10 * np.log10(per)), 0.85 * max(10 * np.log10(per))])
            ax.set_xlim(min(fp), max(fp))
            x, y = fp[np.argmax(per)], np.max(10 * np.log10(per))
            ax.plot(x, y, 'o')
            ax.annotate('%d Hz' % int(x),
                xy=(x, y),
                xytext=(10, 0),
                textcoords='offset points',
                ha='left',
                va='center')

            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Spectrum [$dBm_{50Ω}$] per bin")
            ax.set_title("Measured spectrum for CFO determination")
            plt.show()

        return cfo
