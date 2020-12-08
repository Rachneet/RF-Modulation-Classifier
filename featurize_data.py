import h5py as h5
import numpy as np
import scipy
import csv
from tqdm import tqdm
import pandas as pd

from dataloader import label_idx
from visualize import iq_to_complex
from Receiver import Receiver

dummyReceiver = Receiver(**{'bin_dir': '/home/rachneet/rf_featurized/',
                            'freq': 5750000000.0,
                            'srate': 50000000.0,
                            'name': 'dummy',
                            'rx_dir': '/home/rachneet/rf_featurized/',
                            'bw_noise': 5000000.0,
                            'bw': 100000.0,
                            'vbw': 10000.0,
                            'init_gain': 0,
                            'freq_noise': np.array([5.73e9, 5.77e9]),
                            'span': 50000000.0,
                            'max_gain': 0,
                            'freq_noise_offset': 20000000.0,
                            'bw_signal': 20000000.0})

dummyReceiver.name = 'dummy'

# Parameters
T_s = 1/dummyReceiver.srate # sampling time
A_t = 1 # threshold
N_s = 1024 # window length FFT


def featurize(df, i, row):
    # 24 modulation classes
    classes = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
               'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
               '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
               'AM-DSB-WC', 'OOK', '16QAM']

    r = np.array(row)
    r_star = np.conj(r)

    N = r.size
    x = np.arange(N)

    # zero-mean
    r10 = r - np.mean(r)
    r11 = r_star - np.mean(r_star)

    # zero-crossings
    zc_real = (np.sign(r.real) * np.roll(np.sign(r.real), 1)) < 0
    zc_imag = (np.sign(r.imag) * np.roll(np.sign(r.imag), 1)) < 0
    zc = np.sum(zc_imag) + np.sum(zc_real)

    # Amplitude
    A = np.abs(r)  # amplitude
    mu_A = np.mean(A)  # mean amplitude
    A_N = A / mu_A  # normalized amplitude
    x_w = x[A_N <= A_t]  # indices of weak samples
    x_nw = x[A_N > A_t]  # indices of non-weak samples
    # indices of non-weak samples of 2nd degree (previous sample also non-weak)
    x_nw2 = x_nw[x_nw - np.roll(x_nw, 1) == 1]
    N_c = x_nw.size  # number of non-weak samples
    N_c2 = x_nw2.size  # number of non-weak samples of 2nd degree
    A_CN = A_N - 1  # centered normalized amplitude
    A_CN_nw = A_CN[x_nw]  # centered normalized amplitude of non-weak samples

    # PSD of Amplitude
    fs_A, F1_A_CN = dummyReceiver.calculate_spectrum(data=A_CN, n_fft=N_s)
    F1_A_CN = np.abs(F1_A_CN)
    fs_A, F2_A_CN = dummyReceiver.calculate_spectrum(data=A_CN ** 2, n_fft=N_s)
    F2_A_CN = np.abs(F2_A_CN)
    fs_A, F4_A_CN = dummyReceiver.calculate_spectrum(data=A_CN ** 4, n_fft=N_s)
    F4_A_CN = np.abs(F4_A_CN)

    # Phase
    # my interpretation
    phi = np.angle(r)
    phi_C = phi - np.mean(phi[x_nw])

    # the literature interpretation
    phi_uw = np.unwrap(phi)
    phi_NL = phi_uw  # no detrending because cfo is corrected
    phi_CNL = phi_NL - np.mean(phi_NL[x_nw])

    # Frequency
    f = (1 / 2 * np.pi) * (phi_NL - np.roll(phi_NL, 1)) / T_s
    mu_f = np.mean(f[x_nw2])
    f_CN = (f - mu_f) / (1 / T_s)

    # Spectrogram
    fs, per = dummyReceiver.calculate_spectrum(data=r, n_fft=N_s)
    R = per[abs(fs) <= dummyReceiver.bw_signal / 2] / N_s
    fs = fs[abs(fs) <= dummyReceiver.bw_signal / 2]

    # autocorrelation
    Psi = (np.correlate(r, np.roll(r[:-500], 50), mode='valid') / np.var(r) / N)
    # include noise in sigma!!!

    # Moments

    r20 = r10 * r10
    r21 = r10 * r11
    r22 = r11 * r11
    r40 = r20 * r20

    M20 = (1 / N) * np.sum(r20)
    M21 = (1 / N) * np.sum(r21)
    M22 = (1 / N) * np.sum(r22)
    M40 = (1 / N) * np.sum(r40)
    M41 = (1 / N) * np.sum(r20 * r21)
    M42 = (1 / N) * np.sum(r20 * r22)
    M60 = (1 / N) * np.sum(r40 * r20)
    M61 = (1 / N) * np.sum(r40 * r21)
    M62 = (1 / N) * np.sum(r40 * r22)
    M63 = (1 / N) * np.sum(r20 * r10 * r22 * r11)

    ########################################################################################
    # Features
    # Signal Features
    df.loc[i, "$N_c$ (1st)"] = N_c
    df.loc[i, "$N_c$ (2nd)"] = N_c2
    df.loc[i, "$ZC$"] = zc

    # Spectral Features
    df.loc[i, "$\gamma_{1,max}$"] = np.max(F1_A_CN) ** 2 / N_s
    df.loc[i, "$\gamma_{2,max}$"] = np.max(F2_A_CN) ** 2 / N_s
    df.loc[i, "$\gamma_{4,max}$"] = np.max(F4_A_CN) ** 2 / N_s

    df.loc[i, "$\sigma_{aa}$"] = np.std(np.abs(A_CN))
    df.loc[i, "$\sigma_{a}$"] = np.std(A_CN[x_nw])
    df.loc[i, "$\sigma_{ap,CNL}$"] = np.std(np.abs(phi_CNL[x_nw]))
    df.loc[i, "$\sigma_{dp,CNL}$"] = np.std(phi_CNL[x_nw])
    df.loc[i, "$\sigma_{ap,C}$"] = np.std(np.abs(phi_C[x_nw]))
    df.loc[i, "$\sigma_{dp,C}$"] = np.std(phi_C[x_nw])
    df.loc[i, "$\sigma_{af}$"] = np.std(np.abs(f_CN[x_nw2]))

    df.loc[i, "$\mu^A_{42}$"] = scipy.stats.kurtosis(A_CN, fisher=False)
    df.loc[i, "$\mu^f_{42}$"] = scipy.stats.kurtosis(f_CN[x_nw2], fisher=False)

    df.loc[i, "$\Psi_{max}$"] = np.max(np.abs(Psi))
    df.loc[i, "$\sigma_{R}$"] = np.std(R)
    df.loc[i, "$\mu^R_{42}$"] = scipy.stats.kurtosis(R, fisher=False)
    df.loc[i, "$\\beta$"] = np.sum(r.real ** 2) / np.sum(r.imag ** 2)
    df.loc[i, "$C$"] = np.max(A) / np.sqrt(np.abs(M21))
    df.loc[i, "PAPR"] = np.abs(np.max(r21)) / np.abs(M21)

    # Statistical Features
    df.loc[i, "$\hat{C}_{20}$"] = np.abs(M20)
    df.loc[i, "$\hat{C}_{21}$"] = np.abs(M21)

    df.loc[i, "$\\nu_{42}$"] = np.abs(M42 / np.abs(M21))

    df.loc[i, "$|\widetilde{C}_{40}|$"] = np.abs((-3 * M20 ** 2 + M40) / np.abs(M21) ** 2)
    df.loc[i, "$|\widetilde{C}_{41}|$"] = np.abs((-3 * M21 * M20 + M41) / np.abs(M21) ** 2)
    df.loc[i, "$\widetilde{C}_{42}$"] = np.abs((-2 * M21 ** 2 - M22 * M20 + M42) / np.abs(M21) ** 2)

    df.loc[i, "$|\widetilde{C}_{60}|$"] = np.abs((30 * M20 ** 3 - 15 * M20 * M40 + M60) / np.abs(M21) ** 3)
    df.loc[i, "$|\widetilde{C}_{61}|$"] = np.abs(
        (30 * M20 ** 2 * M21 - 14 * M20 * M41 - M40 * M21 + M61) / np.abs(M21) ** 3)
    df.loc[i, "$|\widetilde{C}_{62}|$"] = np.abs(
        (24 * M21 ** 2 * M20 + 6 * M22 * M20 ** 2 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40 + M62) / np.abs(M21) ** 3)
    df.loc[i, "$\widetilde{C}_{63}$"] = np.abs(
        (12 * M21 ** 3 + 12 * M22 * M21 * M20 - 9 * M42 * M21 + M63) / np.abs(M21) ** 3)


def filter_from_csv(path):
    df = pd.read_csv(path)
    mods = [0,1,2,3]
    df = df.loc[df.label in mods]
    print(df.shape)
    print(df.head())



def main():
    dataset = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
    h5fr = h5.File(dataset, 'r')
    dset1 = list(h5fr.keys())[0]
    dset2 = list(h5fr.keys())[1]
    dset3 = list(h5fr.keys())[2]
    iq = h5fr[dset1][:]
    label = h5fr[dset2][:]
    snr = h5fr[dset3][:]
    df = pd.DataFrame()
    # df['SNR'] = [item for sublist in snr for item in sublist]
    df['SNR'] = [val for val in snr]
    df['label'] = [label_idx(sublist) for sublist in label]

    for i,row in enumerate(tqdm(iq)):
        featurize(df, i, iq_to_complex(row))
        # print(row)

    df.to_csv("/home/rachneet/rf_featurized/dataset_vsg_all_featurized_set.csv", encoding='utf-8', index=False)

    # df2 = pd.read_csv("/home/rachneet/rf_featurized/deepsig_featurized_set.csv", encoding='utf-8')
    # print(df2.head())


if __name__=="__main__":
    main()
    # filter_from_csv("/home/rachneet/rf_featurized/deepsig_featurized_set.csv")