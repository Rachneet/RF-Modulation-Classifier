import h5py as h5
import numpy as np
import random


def read_hdf5(output_path):

    f =  h5.File(output_path,'r')
    m = f['iq']
    l = f['labels']
    s = f['snrs']
    return m,l,s


def shuffle(*datas):

    for d in datas:
        random.seed(444)
        random.shuffle(d)