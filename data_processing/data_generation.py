import os
import glob
import numpy as np

from data_processing.helper_functions import read_binary
from data_processing.Receiver import Receiver


def generate_dataset():

    root_folder = "/data/all_RFSignal_cdv/mixed_recordings/interference/ota"
    param_folder = "/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM"
    output_folder = "npz/"
    # measured_snr = ["0db","5db","10db","15db","20db"]
    mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                   "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"] 

    dummyReceiver = Receiver(**{
                                'freq': 5750000000.0,
                                'srate': 50000000.0,
                                'rx_dir': None,
                                'bw_noise': 5000000.0,
                                'bw': 100000.0,
                                'vbw': 10000.0,
                                'init_gain': 0,
                                'freq_noise': np.array([5.73e9, 5.77e9]),
                                'span': 50000000.0,
                                'max_gain': 0,
                                'freq_noise_offset': 20000000.0,
                                'bw_signal': 20000000.0})

    dummyReceiver.name = "dummy"

    for subdir, dirs, files in os.walk(root_folder+param_folder):
        file_register = {}   # dict to store the folder name as keys and file names as values
        total_files = 0
        for directory in dirs:

            if directory == "bin":
                files_in_directory = glob.glob(os.path.join(subdir, directory) + "/*.bin")
                subfolder = os.path.join(subdir, directory) + "/"
                file_register[subfolder] = files_in_directory
                total_files += len(files_in_directory)

        for iteration, subfolder in enumerate(file_register.keys()):
            print(subfolder)

            #snr = re.search(r"\d+(\.\d+)?", subfolder)
            #print(snr.group(0))
            for mod_scheme in mod_schemes:
                files = [file for file in file_register[subfolder] if mod_scheme in file]
                num_files = len(files)
                # print(num_files)
                if (num_files > 0):
                    print("OS", "Analyzing samples from %s modulation" % mod_scheme)

                    # find dimensionality of output
                    num_samples = num_files * 8  # each file = 10 sample record, discard beginning and end
                    num_iq = int(read_binary(files[0]).size / 10)
                    print("OS", "%d samples with %d IQ samples each" % (num_samples, num_iq))

                    # pre-allocate memory
                    matrix = np.zeros((num_samples, num_iq), dtype=np.complex64)
                    snrs = -1 * np.ones(num_samples, dtype=float)
                    labels = -1 * np.ones(num_samples, dtype=int)

                    # for every binary file
                    for i, file in enumerate(files):
                        # read data, load in dummy receiver
                        data = read_binary(file)
                        dummyReceiver.data = data

                        # get SNR, label, and filter data
                        snrs[i * 8:(i + 1) * 8] = dummyReceiver.measure_SNR()                  #int(snr.group(0))
                        labels[i * 8:(i + 1) * 8] = mod_schemes.index(mod_scheme)
                        filtered_data = dummyReceiver.filter_data()

                        # for every chunk in a binary file (only use chunk 2-9)
                        # one chunk is equal to one measurement, observation or sample in the ML sense
                        # for this configuration every observation is t=0.001s (1ms) long
                        matrix[i * 8:(i + 1) * 8, :] = np.stack(np.split(filtered_data, 10)[1:-1])

                    # save data after all files have been read
                    output_dir = subfolder[:-len("bin/")] + output_folder
                    output_file = output_dir + mod_scheme + "_filtered.npz"

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    np.savez(output_file, matrix=matrix, snrs=snrs, labels=labels)
                    print("OS", "saved .npz data in %s" % output_file)