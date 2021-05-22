import pandas as pd
import numpy as np
import h5py as h5
import os
import glob
from tqdm import tqdm

import data_processing.read_h5 as reader
from data_processing.read_filtered import sort_matrix_entries, encode_labels
from data_processing.dataloader import label_idx, create_set


def exrapolate_data():
    """
    function to generate more samples from iq segments
    :return: None
    """

    root_folder = "/media/backup/Arsenal/rf_dataset_inets/"
    param_folder = "/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM"
    output_path = root_folder + "dataset_intf_free_vsg_sc_snr20.h5"
    file_paths = []

    for subdir, dirs, files in os.walk(root_folder + param_folder):
        for directory in dirs:
            if directory == "npz":
                files_in_directory = glob.glob(os.path.join(subdir, directory) + "/*.npz")
                file_paths.extend((files_in_directory))

    print("Total number of files: ", len(file_paths))

    num_iq = 50000
    batch = 1024
    samples_per_segment = int(num_iq / batch)
    cut_off = samples_per_segment * batch

    for i, path in enumerate(file_paths):

        data = np.load(path)
        iq = data['matrix'][:, :cut_off]
        labels = data['labels']
        snrs = data['snrs']
        total_samples = labels.shape[0] * int(num_iq / batch)  # samples after batching iq to 128
        # batch_iq = np.array([np.array(np.split(iq[row],samples_per_segment)) for row in range(len(iq))])
        batch_labels = np.zeros(samples_per_segment, dtype=np.int8)
        batch_snrs = np.zeros(samples_per_segment, dtype=np.int8)

        for row in range(len(iq)):
            batch_iq = np.array(np.split(iq[row], samples_per_segment))
            batch_labels[:] = np.array(labels[row])
            batch_snrs[:] = np.array(snrs[row])

            processed_iq = np.apply_along_axis(sort_matrix_entries, axis=1, arr=batch_iq)
            processed_labels = encode_labels(8, batch_labels)

            if not os.path.exists(output_path):
                with h5.File(output_path, 'w') as hdf:
                    hdf.create_dataset('iq', data=processed_iq, chunks=True, maxshape=(None, batch, 2),
                                       compression='gzip')
                    hdf.create_dataset('labels', data=processed_labels, chunks=True, maxshape=(None, 8),
                                       compression='gzip')
                    hdf.create_dataset('snrs', data=batch_snrs, chunks=True, maxshape=(None,), compression='gzip')
                    print(hdf['iq'].shape)

            else:
                with h5.File(output_path, 'a') as hf:
                    hf["iq"].resize((hf["iq"].shape[0] + batch_iq.shape[0]), axis=0)
                    hf["iq"][-batch_iq.shape[0]:] = processed_iq
                    hf["labels"].resize((hf["labels"].shape[0] + batch_labels.shape[0]), axis=0)
                    hf["labels"][-batch_labels.shape[0]:] = processed_labels
                    hf["snrs"].resize((hf["snrs"].shape[0] + batch_snrs.shape[0]), axis=0)
                    hf["snrs"][-batch_snrs.shape[0]:] = batch_snrs
                    print(hf['iq'].shape)


def sample_signals_from_dataset(data_path, out_path):
    """
    Sample signals from complete dataset
    :param data_path: [str] path to data to be sampled
    :param out_path: [str] save path for the sampled set
    :return: None
    """
    iq, labels, snrs = reader.read_hdf5(data_path)
    # print(len(labels))
    # count = 0
    # for l in labels:
    #     if label_idx(l) == 0:
    #         count += 1
    # print(count)
    threshold = int(0.2 * 153600)
    mods = [0, 1, 2, 3, 4, 5, 6, 7]
    for mod in mods:
        count0, count5, count10, count15, count20 = 0, 0, 0, 0, 0
        for i in tqdm(range(iq.shape[0])):
            if label_idx(labels[i]) == mod:
                if snrs[i] == 0:
                    if count0 < threshold:
                        create_set(out_path, iq[i], labels[i], snrs[i])
                        count0 += 1

                elif snrs[i] == 5:
                    if count5 < threshold:
                        create_set(out_path, iq[i], labels[i], snrs[i])
                        count5 += 1

                elif snrs[i] == 10:
                    if count10 < threshold:
                        create_set(out_path, iq[i], labels[i], snrs[i])
                        count10 += 1

                elif snrs[i] == 15:
                    if count15 < threshold:
                        create_set(out_path, iq[i], labels[i], snrs[i])
                        count15 += 1

                elif snrs[i] == 20:
                    if count20 < threshold:
                        create_set(out_path, iq[i], labels[i], snrs[i])
                        count20 += 1
        print(count0, count5, count10, count15, count20)


if __name__ == '__main__':
    path = "/home/rachneet/rf_dataset_inets/interference_free/no_cfo/usrp/usrp_no_intf_all_normed.h5"
    out_path = "/home/rachneet/rf_dataset_inets/usrp_no_intf_20_percent_sampled.h5"
    sample_signals_from_dataset(path, out_path)