import scipy.signal
import numpy as np


def filter_samples(data):
    # filter config
    numtaps = 100
    srate = 50000000
    bw_signal = 20000000

    # using a low pass filter
    b = scipy.signal.firwin(numtaps=numtaps,
                                cutoff=bw_signal / 2,
                                pass_zero=True,
                                fs=srate
                                )

    filtered_data = scipy.signal.convolve(data,b,mode='same')

    return filtered_data
