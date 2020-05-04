import os
import glob
import sys
import numpy as np
import struct
import filter_data
import re

from helper_functions import info,read_binary
from filter_data import filter_samples
from Receiver import Receiver
print(os.getcwd())
# print(os.listdir("/data"))
# file = "/data/all_RFSignal_cdv/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_BPSK/10db/sir_5db/bin/SC_BPSK_1.bin"
# size = os.stat(file).st_size
# print("file size: ", size)
#
# samples = int(size/8)
# print("number of samples: ", samples)
#  # one file contains a single recording of 10 ms
#  # 1 ms at front and end are removed to account for transient effects
#  # thus 8 ms recording
#
#
# data = np.array([0 + 0 *1j] * samples, dtype = np.complex64)
#
# with open(file, 'rb') as binary_file:
#     binary_data = binary_file.read()
#     unpacked_data = np.array(struct.unpack('f' * samples * 2, binary_data))
#     data.real = unpacked_data[::2]
#     data.imag = unpacked_data[1::2]
#
# print("real data: ",data.real)
# print("Img data: ",data.imag)
# print(len(data.real))
# print(len(data.imag))
#
# num_iq = data.size/10  # per signal segment
# print("Number of IQ samples",num_iq)
#
# filter_data = filter_data.filter_samples(data)
# # print(filter_data)
# # print(len(filter_data))
# # print(filter_data.shape)
# # print(filter_data.real)
#
# i=0
# # allocate memory
# matrix = np.zeros((8, num_iq), dtype=np.complex64)
# # remove transients: first and last segments
# # 8 segments in total each of 1ms
# matrix[i*8:(i+1)*8, :] = np.stack(np.split(filter_data, 10)[1:-1])
#
# print("shape of matrix:",matrix.shape)
# print(matrix[0])

#print(np.array((matrix.real,matrix.imag)))

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
        #print(subdir)
        #print(dirs)
        # print(files)
        file_register = {}   # dict to store the folder name as keys and file names as values
        total_files = 0
        for directory in dirs:

            if directory == "bin":
                #print(glob.glob(os.path.join(subdir, directory)))
                files_in_directory = glob.glob(os.path.join(subdir, directory) + "/*.bin")
                subfolder = os.path.join(subdir, directory) + "/"
                file_register[subfolder] = files_in_directory
                total_files += len(files_in_directory)

        #print("sys", "Found %d files in %d directories" % (total_files, len(file_register.keys())))

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








