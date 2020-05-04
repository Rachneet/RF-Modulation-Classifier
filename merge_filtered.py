import numpy as np
import os
import glob
import h5py as h5
import random
from read_filtered import sort_matrix_entries, encode_labels
from sklearn import preprocessing
import dataloader as dl
import pandas as pd

def merge_files():

    # file1_path = "/data/all_RFSignal_cdv/mixed_recordings/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM/npz/OFDM_QPSK_filtered.npz"
    # file2_path = '/data/all_RFSignal_cdv/mixed_recordings/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM/npz/OFDM_16QAM_filtered.npz'
    # x = np.load(file1_path)
    # d = x['matrix']
    # l = x['labels']
    # s = x['snrs']
    # # i=0
    # # for row in d:
    # #     print(row)
    # #     i+=1
    # #     if i==3:
    # #         break
    # # print(d.shape[0])
    # h5_path = '/data/all_RFSignal_cdv/mixed_recordings/interference/ota/out.h5'
    # new_arr = np.apply_along_axis(sort_matrix_entries, axis=1, arr=d)
    # print(new_arr.shape)

    #if not os.path.exists(h5_path):

    # with h5.File(h5_path,'w') as hdf:
    #     hdf.create_dataset('iq',data=new_arr,chunks=True,maxshape=(None,50000,2),compression='gzip')
    #     hdf.create_dataset('labels', data=l, chunks=True, maxshape=(None,), compression='gzip')
    #     hdf.create_dataset('snrs', data=s, chunks=True, maxshape=(None,), compression='gzip')
    #
    # with h5.File(h5_path,'a') as hf:
    #     hf["iq"].resize((hf["iq"].shape[0] + d.shape[0]), axis=0)
    #     hf["iq"][-d.shape[0]:] = new_arr
    #     hf["labels"].resize((hf["labels"].shape[0] + l.shape[0]), axis=0)
    #     hf["labels"][-l.shape[0]:] = l
    #     hf["snrs"].resize((hf["snrs"].shape[0] + s.shape[0]), axis=0)
    #     hf["snrs"][-s.shape[0]:] = s
    #     print(hf['iq'].shape)
    #     print(hf['iq'][:2])
    #     print(hf['labels'].shape)
    #     print(hf['labels'][:2])
    #     print(hf['snrs'].shape)
    #     print(hf['snrs'][:2])

    root_folder = "/media/backup/Arsenal/interference_new/interference/ota"
    param_folder = "/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM"
    output_path = root_folder + "/dataset_intf_sc_16QAM.h5"
    file_paths = []

    for subdir, dirs, files in os.walk(root_folder+param_folder):
        #print(dirs)
        for directory in dirs:
            if directory == "npz":
                files_in_directory = glob.glob(os.path.join(subdir, directory) + "/*.npz")
                # subfolder = os.path.join(subdir, directory) + "/"
                # print(subfolder)
                # print(files_in_directory)
                file_paths.extend((files_in_directory))
    # print(file_paths)
    # print(len(file_paths))
    # sorted_paths =[]
    # for name in file_paths:
    #      if "intf_vsg/i_OFDM_64QAM" in name:
    #          sorted_paths.append(name)
    print("Total number of files: ",len(file_paths))
    # # print(sorted_paths)
    #
    num_samples = []
    num_iq = 50000
    total_samples = 0
    for path in file_paths:
        data = np.load(path)
        label = data['labels']
        num_samples.append(len(label))
    total_samples = sum(num_samples)
    print("Samples accounted for: {}".format(total_samples))
    print("Starting merge...")
    #
    for i,path in enumerate(file_paths):
        data = np.load(path)
        iq = data['matrix']
        labels = data['labels']
        snrs = data['snrs']

        processed_iq = np.apply_along_axis(sort_matrix_entries, axis=1, arr=iq)
        processed_labels = encode_labels(8,labels)

        if not os.path.exists(output_path):
            with h5.File(output_path,'w') as hdf:
                hdf.create_dataset('iq',data=processed_iq,chunks=True,maxshape=(None,num_iq,2),compression='gzip')
                hdf.create_dataset('labels', data=processed_labels, chunks=True, maxshape=(None,8), compression='gzip')
                hdf.create_dataset('snrs', data=snrs, chunks=True, maxshape=(None,), compression='gzip')
                print(hdf['iq'].shape)

        else:
            with h5.File(output_path, 'a') as hf:
                hf["iq"].resize((hf["iq"].shape[0] + iq.shape[0]), axis=0)
                hf["iq"][-iq.shape[0]:] = processed_iq
                hf["labels"].resize((hf["labels"].shape[0] + labels.shape[0]), axis=0)
                hf["labels"][-labels.shape[0]:] = processed_labels
                hf["snrs"].resize((hf["snrs"].shape[0] + snrs.shape[0]), axis=0)
                hf["snrs"][-snrs.shape[0]:] = snrs
                print(hf['iq'].shape)



        print("Files merged:", i+1)




    #     if i>0:
    #         merged_arr = np.vstack((merged_arr,mat))
    #         print(merged_arr.shape )
    #     elif i==0:
    #         merged_arr = mat
    #         print(merged_arr.shape)
    #     print(i)
    #     i = i+1
    # merged_arr = np.vstack(arr_list)
    # print(merged_arr.shape)



    # file1_path = "/data/all_RFSignal_cdv/mixed_recordings/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM/npz/OFDM_QPSK_filtered.npz"
    # file2_path = '/data/all_RFSignal_cdv/mixed_recordings/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_SC_16QAM/npz/OFDM_16QAM_filtered.npz'
    # f1 = np.load(file1_path)
    #
    # f2 = np.load(file2_path)
    # mat1 = f1['matrix']
    # mat2 = f2['matrix']
    # l1 = f1['labels']
    # print(len(mat1))
    # p = np.random.permutation(len(mat1))
    # print(mat1[p], l1[p])
    #
    # print(mat1.shape,mat2.shape)
    # new_mat = np.vstack((mat1,mat2))
    # print(new_mat.shape)
    # print(new_mat)

def merge_hdfset():
    path = "/media/backup/Arsenal/rf_dataset_inets/"
    path_deepsig = "/media/backup/Arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/"
    # files = [path+"dataset_intf_free_no_cfo_vsg_snr0_1024.h5",
    #          path+"dataset_intf_free_no_cfo_vsg_snr5_1024.h5",
    #          path+"dataset_intf_free_no_cfo_vsg_snr10_1024.h5",
    #          path+"dataset_intf_free_no_cfo_vsg_snr15_1024.h5",
    #          path+"dataset_intf_free_no_cfo_vsg_snr20_1024.h5"]
    files = [path + "dataset_intf_free_no_cfo_vsg_snr20_1024.h5",
             path_deepsig + "GOLD_XYZ_OSC.0001_1024.hdf5"]
    mods_skip = [8,9,18,19,23]
    with h5.File(path+'dataset_un_vsg_snr20.h5', mode='w') as file:
        row1 = 0
        i = 0
        for dataset in files:
            h5fr = h5.File(dataset, 'r')
            dset1 = list(h5fr.keys())[0]
            dset2= list(h5fr.keys())[1]
            dset3 = list(h5fr.keys())[2]
            iq_data = h5fr[dset1][:]
            label_data = h5fr[dset2][:]
            snr_data = h5fr[dset3][:]

            if dataset[-4:] == "hdf5":
                print("in hdf5")
                label_idx = [dl.label_idx(label) for label in label_data]
                idx_class = [i for i in range(len(label_idx)) if label_idx[i] not in mods_skip]

                iq_data = iq_data[idx_class]
                # label_data[idx_class] = np.array([0,0,0,0,0,0,0,0,1])
                label_data = np.array([np.array([0,0,0,0,0,0,0,0,1]) for _ in label_data[idx_class]])
                snr_data = snr_data[idx_class]
                snr_data = snr_data.astype('int8')
            else:
                label_data = np.array([np.append(label,np.array([0])) for label in label_data])
                # df = pd.DataFrame()
                # df['iq'] = list(map(lambda x: np.array(x), iq_data))
                # df['normalized_iq'] = df.iq.apply(lambda x: preprocessing.scale(x, with_mean=False))
                # df.drop('iq', axis=1, inplace=True)
                # df['labels'] = list(map(lambda x: np.array(x), label_data))
                # df['snr'] = list(map(lambda x: x, snr_data))
                # df = df.sample(frac=1, random_state=4)

            # normalize data
            norm_iq = [preprocessing.scale(sample, with_mean=False) for sample in iq_data]
            norm_iq = np.array(norm_iq)
            print("norm done")

            dslen = iq_data.shape[0]
            cols_iq = iq_data.shape[1]
            rows_iq = iq_data.shape[2]
            cols_labels = label_data.shape[1]

            # print(dslen,rows_iq,rows_iq,cols_labels,cols_snrs)
            # break
            if row1 == 0:
                file.create_dataset('iq', dtype="f", shape=(dslen, cols_iq, rows_iq), maxshape=(None, cols_iq, rows_iq))
                file.create_dataset('labels', dtype="f", shape=(dslen, cols_labels), maxshape=(None, cols_labels))
                file.create_dataset('snrs', dtype="f", shape=(dslen,), maxshape=(None,))
            if row1 + dslen <= len(file['iq']):
                file['iq'][row1:row1 + dslen, :,:] = norm_iq[:]
                file['labels'][row1:row1 + dslen, :] = label_data[:]
                file['snrs'][row1:row1 + dslen,] = snr_data.squeeze()
            else:
                file['iq'].resize((row1 + dslen, cols_iq, rows_iq))
                file['iq'][row1:row1 + dslen, :,:] = norm_iq[:]
                file['labels'].resize((row1 + dslen, cols_labels))
                file['labels'][row1:row1 + dslen, :] = label_data[:]
                file['snrs'].resize((row1 + dslen, ))
                file['snrs'][row1:row1 + dslen,] = snr_data.squeeze()
            row1 += dslen
            i = i + 1
            print("Datasets processed: {}".format(i))




if __name__=="__main__":
    # merge_hdfset()
    pass