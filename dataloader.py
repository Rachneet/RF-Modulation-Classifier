import read_h5 as reader
import numpy as np
import random
import pandas as pd
import ast
import h5py as h5
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def label_idx(labels):
    for val in range(len(labels)):
        if labels[val] == 1:
            return val


def load_batch(path,batch_size=256,mode="train"):
    print("Loading Data...")

    training_params = {"batch_size": batch_size,
                       "shuffle": False,
                       "num_workers": 4}

    if path[-3:]=="npz":

        data = np.load(path, allow_pickle=True)
        iq = data['matrix']
        labels = data['labels']

        train_bound = int(0.75 * labels.shape[0])
        val_bound = int(0.80 * labels.shape[0])

        x_train_gen = DataLoader(iq[:train_bound], **training_params)
        y_train_gen = DataLoader(labels[:train_bound], **training_params)
        x_val_gen = DataLoader(iq[train_bound:val_bound], **training_params)
        y_val_gen = DataLoader(labels[train_bound:val_bound], **training_params)

        x_train_val = iq[:train_bound]
        y_train_val = labels[:train_bound]

        print("Number of batches : {}".format(len(x_train_gen)))

        return x_train_gen, y_train_gen, x_val_gen, y_val_gen, x_train_val, y_train_val

    else:

        iq,labels,snrs = reader.read_hdf5(path)
        print("=======Starting======")
        # print(iq.shape)
        # print(labels.shape)
        # -----------------------------unk set--------------------------------------------------
        # n =153600
        # mods_skip = [8, 9, 18, 19, 23]
        # dataset = "/media/backup/Arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
        # h5fr = h5.File(dataset, 'r')
        # dset1 = list(h5fr.keys())[0]
        # dset2 = list(h5fr.keys())[1]
        # dset3 = list(h5fr.keys())[2]
        # iq_data = h5fr[dset1][:]
        # label_data = h5fr[dset2][:]
        # # snr_data = h5fr[dset3][:]
        # label_idx_ = [label_idx(label) for label in label_data]
        # idx_class = [i for i in range(len(label_idx_)) if label_idx_[i] not in mods_skip]
        # iq_data = iq_data[idx_class]
        # # label_data[idx_class] = np.array([0,0,0,0,0,0,0,0,1])
        # label_data = np.array([np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in label_data[idx_class]])
        # snr_data = snr_data[idx_class]
        # snr_data = snr_data.astype('int8')
        # -----------------------------------------------------------------------------------------------

        # rand_iq = np.zeros((n,1024, 2), dtype=np.float32)
        # rand_label = np.zeros((n, 9), dtype=np.float32)
        # for i in range(n):
        #     rand_iq[i] = np.random.rand(1024,2)
        #     rand_label[i] = np.array([0,0,0,0,0,0,0,0,1],dtype=np.float32)
        # # print(rand_label[0])
        #
        # df2 = pd.DataFrame()
        # df2['iq'] = list(map(lambda x:np.array(x,dtype=np.float32),iq_data))
        # df2['labels'] = list(map(lambda x: np.array(x,dtype=np.float32), label_data))
        # df2 = df2.sample(frac=1, random_state=4)
        # print("First df done")

        # analyze data
        df = pd.DataFrame()
        # df['iq'] = list(map(lambda x:np.array(x,dtype=np.float32),iq))
        # df['normalized_iq'] = df.iq.apply(lambda x: preprocessing.scale(x, with_mean=False))
        # df.drop('iq', axis=1, inplace=True)
        df['labels'] = list(map(lambda x: np.array(x,dtype=np.float32), labels))
        # df['snr'] = list(map(lambda x: x, snrs))

        # df['labels'] = df.labels.apply(lambda x: np.append(x,np.array([0], dtype=np.float32)))
        # print("second df done")
        # df = df.sample(frac=1, random_state=4)
        # print("====start merge====")
        # combined_df = pd.concat([df,df2[:n]],ignore_index=True)
        # combined_df['normalized_iq'] = combined_df.iq.apply(lambda x: preprocessing.scale(np.array(x), with_mean=False))
        # combined_df.drop('iq', axis=1, inplace=True)
        # combined_df['labels'] = combined_df.labels.apply(lambda x: np.reshape(x,(-1,9)))
        # print("merged")

        train_bound = int(0.75 * df.labels.shape[0])
        val_bound = int(0.80 * df.labels.shape[0])
        test_bound = int(0.85 * df.labels.shape[0])


        df['label_id'] = df['labels'].apply(lambda x: label_idx(x))

        # df = df.groupby('label_id').apply(lambda s: s.sample(5000)).reset_index(drop=True)
        # df.drop('label_id', axis=1, inplace=True)
        print(df.label_id.value_counts())

        # combined_df = combined_df.sample(frac=1,random_state=4)
        # print("sampling done")
        # print(type(combined_df.normalized_iq.values))
        # print(combined_df.normalized_iq.values[0].shape)
        # print(combined_df.labels.values[0].shape)
        # output_path = "/media/backup/Arsenal/rf_dataset_inets/dataset_unknown_vsg_snr20.h5"
        # combined_df.to_hdf(output_path, key='df', mode='w')
        # if not os.path.exists(output_path):
        # with h5.File(output_path,'w') as hdf:
        #     dt = h5.vlen_dtype(np.dtype('float32'))
        #     hdf.create_dataset('iq',data=combined_df.normalized_iq.values,chunks=True,
        #                        maxshape=(1382400,),compression='gzip',dtype=dt)
        #     hdf.create_dataset('labels', data=combined_df.labels.values, chunks=True,
        #                        maxshape=(1382400,), compression='gzip',dtype=dt)
                # hdf.create_dataset('snrs', data=snrs, chunks=True, maxshape=(None,), compression='gzip')
        # combined_df.to_csv('/media/backup/Arsenal/rf_dataset_inets/unk_class_vsg20.csv', header=True, index=False)
        # df.dropna(inplace=True)
        # print(df.isnull().any())
        # df.drop('label_id', inplace= True)

        # matrix = df.normalized_iq.values
        # labels = df.labels.values
        #
        # np.savez("/media/backup/Arsenal/rf_dataset_inets/dataset_vsg_sc_snr20.npz", matrix=matrix, labels=labels)

        # using dataframe values
        if mode=='train':
            x_train_gen = DataLoader(df.normalized_iq[:train_bound].values, **training_params)
            y_train_gen = DataLoader(df.labels[:train_bound].values, **training_params)
            x_val_gen = DataLoader(df.normalized_iq[train_bound:val_bound].values, **training_params)
            y_val_gen = DataLoader(df.labels[train_bound:val_bound].values, **training_params)

            return x_train_gen, y_train_gen, x_val_gen, y_val_gen

        elif mode=='test':
            x_test_gen = DataLoader(df.normalized_iq[val_bound:].values, **training_params)
            y_test_gen = DataLoader(df.labels[val_bound:].values, **training_params)
            y_test_raw = df.labels[val_bound:].values

            return x_test_gen, y_test_gen,y_test_raw

        elif mode=="both":
            x_train_gen = DataLoader(df.normalized_iq[:train_bound].values, **training_params)
            y_train_gen = DataLoader(df.labels[:train_bound].values, **training_params)
            x_val_gen = DataLoader(df.normalized_iq[train_bound:val_bound].values, **training_params)
            y_val_gen = DataLoader(df.labels[train_bound:val_bound].values, **training_params)
            x_test_gen = DataLoader(df.normalized_iq[val_bound:].values, **training_params)
            y_test_gen = DataLoader(df.labels[val_bound:].values, **training_params)
            y_test_raw = df.labels[val_bound:].values
            snr_test_gen = DataLoader(df.snr[val_bound:].values, **training_params)

            return x_train_gen, y_train_gen, x_val_gen, y_val_gen,x_test_gen, y_test_gen,y_test_raw,snr_test_gen
        else:
            pass


from py_lightning import DatasetFromHDF5
import math


def load_data(data_path, val_fraction, test_fraction, **training_params):

    dataset = DatasetFromHDF5(data_path, 'iq', 'labels', 'sirs')
    num_train = len(dataset)
    indices = list(range(num_train))
    val_split = int(math.floor(val_fraction * num_train))
    test_split = val_split + int(math.floor(test_fraction * num_train))

    if not ('shuffle' in training_params and not training_params['shuffle']):
        np.random.seed(4)
        np.random.shuffle(indices)
    if 'num_workers' not in training_params:
        training_params['num_workers'] = 1

    train_idx, valid_idx, test_idx = indices[test_split:], indices[:val_split], indices[val_split:test_split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
                                    shuffle=False, num_workers=training_params['num_workers'],
                                    sampler=train_sampler)
    val_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
                                  shuffle=False, num_workers=training_params['num_workers'],
                                  sampler=valid_sampler)
    test_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
                                   shuffle=False, num_workers=training_params['num_workers'],
                                   sampler=test_sampler)
    return train_dataset, val_dataset, test_dataset


if __name__=="__main__":

    # iq = np.array([[[0.2 , 0.2],[0.2 , 0.2]],
    #  [[0.3 , 0.3],[0.2 , 0.2]],
    #  [[0.4 , 0.4],[0.2 , 0.2]],
    #  [[0.5 , 0.5],[0.2 , 0.2]],
    #  [[0.6,  0.6],[0.2 , 0.2]]])
    # print(iq.shape)
    # labels = np.array([[1,0,0,0],[0,1,0,0],
    #           [0,0,1,0],[0,0,0,1],[0,1,0,0]])
    # df = pd.DataFrame()
    # # df['iq'] = list(map(lambda x: np.array(x), iq))
    # # df['labels'] = list(map(lambda x: np.array(x), labels))
    # x =[[0, 0, 0, 0, 0, 0, 1, 0, 0]]
    # print(np.array(x))
    # true_label = label_idx(x[0])
    # print(true_label)
    # with h5.File("test.h5", 'w') as hdf:
    #     dt = h5.special_dtype(vlen=np.dtype('float32'))
    #     hdf.create_dataset('iq', data=df.iq.values, chunks=True,
    #                        maxshape=(5,), compression='gzip', dtype=dt)
    # # df.to_hdf('test.h5', key='xyz', mode='w')
    #
    # data = h5.File('test.h5','r')
    # x = data['iq']
    # print(x[0])
    # print(x[0].shape)
    # for sample in x:
    #     print(sample)
    # print(type(df.iq.values))
    #
    # print(df.head())
    # print(len(df.iq.values))
    # print(label_idx(df.labels.values[1]))
    #
    # matrix = np.zeros((5,2), dtype=np.complex64)
    # matrix = df.iq.values
    # print(matrix)
    # print(matrix[0].shape)
    #
    # normalize_iq(iq)

    # path_free = "/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_vsg_cfo1_all.h5"
    # # path = "/media/backup/Arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
    # data = h5.File(path_free,'r')
    # iq,labels,snrs = data['iq'],data['labels'],data['snrs']

    # print(iq.shape)
    # print(labels.shape)
    # label_data = np.array([np.array([0,0,0,0,0,0,0,0,1]) for label in labels[:100]])
    # print(label_data)
    # print(label_data.shape)
    # label_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    # print(label_data.shape)
    load_batch("/media/backup/Arsenal/rf_dataset_inets/dataset_intf_bpsk_usrp_snr20_sir25_1024.h5",mode='')

