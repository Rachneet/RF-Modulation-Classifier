import pandas as pd
import ast
import h5py as h5
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from models.pytorch_lightning.py_lightning import *
import math
from tqdm import tqdm
from multiprocessing import Pool


def label_idx(labels):
    for val in range(len(labels)):
        if labels[val] == 1:
            return val


def parallelize_dataframe(df, func, n_cores=10):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def load_batch(path,batch_size=512,mode="train"):
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

        # analyze data
        df = pd.DataFrame()
        df['iq'] = list(map(lambda x:np.array(x,dtype=np.float32),iq))
        # df['normalized_iq'] = df.iq.apply(lambda x: preprocessing.scale(x, with_mean=False))
        # df.drop('iq', axis=1, inplace=True)
        df['labels'] = list(map(lambda x: np.array(x,dtype=np.float32), labels))
        df['snr'] = list(map(lambda x: x, snrs))

        df = df.sample(frac=1, random_state=4)
        train_bound = int(0.75 * df.labels.shape[0])
        val_bound = int(0.80 * df.labels.shape[0])

        # using dataframe values
        if mode == 'train':
            x_train_gen = DataLoader(df.iq[:train_bound].values, **training_params)
            y_train_gen = DataLoader(df.labels[:train_bound].values, **training_params)
            x_val_gen = DataLoader(df.iq[train_bound:val_bound].values, **training_params)
            y_val_gen = DataLoader(df.labels[train_bound:val_bound].values, **training_params)

            return x_train_gen, y_train_gen, x_val_gen, y_val_gen

        elif mode == 'test':
            x_test_gen = DataLoader(df.normalized_iq[val_bound:].values, **training_params)
            y_test_gen = DataLoader(df.labels[val_bound:].values, **training_params)
            y_test_raw = df.labels[val_bound:].values

            return x_test_gen, y_test_gen,y_test_raw

        elif mode == "both":
            x_train_gen = DataLoader(df.iq[:train_bound].values, **training_params)
            y_train_gen = DataLoader(df.labels[:train_bound].values, **training_params)
            x_val_gen = DataLoader(df.iq[train_bound:val_bound].values, **training_params)
            y_val_gen = DataLoader(df.labels[train_bound:val_bound].values, **training_params)
            x_test_gen = DataLoader(df.iq[val_bound:].values, **training_params)
            y_test_gen = DataLoader(df.labels[val_bound:].values, **training_params)
            y_test_raw = df.labels[val_bound:].values
            snr_test_gen = DataLoader(df.snr[val_bound:].values, **training_params)

            return x_train_gen, y_train_gen, x_val_gen, y_val_gen,x_test_gen, y_test_gen,y_test_raw,snr_test_gen
        else:
            pass


def load_data(data_path, val_fraction, test_fraction, **training_params):

    print("=============Loading Data==================")
    dataset = DatasetFromHDF5(data_path, 'iq', 'labels', 'snrs')
    print("=============Data Loaded====================")
    num_train = len(dataset)
    indices = list(range(num_train))
    val_split = int(math.floor(val_fraction * num_train))
    test_split = val_split + int(math.floor(test_fraction * num_train))

    if not ('shuffle' in training_params and not training_params['shuffle']):
        np.random.seed(4)
        np.random.shuffle(indices)
    if 'num_workers' not in training_params:
        training_params['num_workers'] = 1

    test_idx = indices[val_split:test_split]
    # train_idx, valid_idx, test_idx = indices[test_split:], indices[:val_split], indices[val_split:test_split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # train_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
    #                                 shuffle=False, num_workers=training_params['num_workers'],
    #                                 sampler=train_sampler)
    # val_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
    #                               shuffle=False, num_workers=training_params['num_workers'],
    #                               sampler=valid_sampler)
    test_dataset = DataLoader(dataset, batch_size=training_params['batch_size'],
                                   shuffle=False, num_workers=training_params['num_workers'],
                                   sampler=test_sampler)
    print("==================Dataset created and batched==========================")
    # return train_dataset, val_dataset, test_dataset
    return test_dataset


def create_set(path, iq, label, sir):
    iq = iq.reshape(-1,1024,2)
    label = label.reshape(-1,8)
    sir = sir.reshape(-1)
    if not os.path.exists(path):
        with h5.File(path, 'w') as hdf:
            hdf.create_dataset('iq', data=iq, chunks=True, maxshape=(None, 1024, 2),
                               compression='gzip')
            hdf.create_dataset('labels', data=label, chunks=True, maxshape=(None, 8),
                               compression='gzip')
            hdf.create_dataset('sirs', data=sir, chunks=True, maxshape=(None,),
                               compression='gzip')
    else:
        with h5.File(path, 'a') as hf:
            hf["iq"].resize((hf["iq"].shape[0] + iq.shape[0]), axis=0)
            hf["iq"][-iq.shape[0]:] = iq
            hf["labels"].resize((hf["labels"].shape[0] + label.shape[0]), axis=0)
            hf["labels"][-label.shape[0]:] = label
            hf["sirs"].resize((hf["sirs"].shape[0] + sir.shape[0]), axis=0)
            hf["sirs"][-sir.shape[0]:] = sir


def sequential_set():
    path = "/media/rachneet/arsenal/rf_dataset_inets/dataset_intf_ofdm_snr10_1024.h5"
    train_path = "/media/rachneet/arsenal/rf_dataset_inets/sequential_sets/train90_sequential_intf_ofdm.h5"
    test_path = "/media/rachneet/arsenal/rf_dataset_inets/sequential_sets/test10_sequential_intf_ofdm.h5"
    data = h5.File(path, 'r')
    iq, labels, sirs = data['iq'], data['labels'], data['sirs']
    # print(labels.shape)
    threshold1 = int(0.1*57600)
    threshold2 = int(0.1*48000)
    mods = [0,1,2,3,4,5,6,7]
    for mod in mods:
        count5, count10, count15, count20, count25 = 0, 0, 0, 0, 0
        for i in tqdm(range(iq.shape[0])):
            if label_idx(labels[i]) == mod:
                if sirs[i] == 5:
                    if count5 < threshold1:
                        create_set(test_path, iq[i], labels[i], sirs[i])
                        count5 += 1
                    else:
                        create_set(train_path, iq[i], labels[i], sirs[i])
                elif sirs[i] == 10:
                    if count10 < threshold1:
                        create_set(test_path, iq[i], labels[i], sirs[i])
                        count10 += 1
                    else:
                        create_set(train_path, iq[i], labels[i], sirs[i])
                elif sirs[i] == 15:
                    if count15 < threshold1:
                        create_set(test_path, iq[i], labels[i], sirs[i])
                        count15 += 1
                    else:
                        create_set(train_path, iq[i], labels[i], sirs[i])
                elif sirs[i] == 20:
                    if count20 < threshold1:
                        create_set(test_path, iq[i], labels[i], sirs[i])
                        count20 += 1
                    else:
                        create_set(train_path, iq[i], labels[i], sirs[i])
                elif sirs[i] == 25:
                    if count25 < threshold2:
                        create_set(test_path, iq[i], labels[i], sirs[i])
                        count25 += 1
                    else:
                        create_set(train_path, iq[i], labels[i], sirs[i])
    print(count5, count10, count15, count20, count25)


if __name__=="__main__":
    load_batch("/media/rachneet/arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
                      , 512, mode='train')