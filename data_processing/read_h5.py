import h5py as h5
import numpy as np
import random

# root_folder = "/data/all_RFSignal_cdv/interference/ota"
# output_path = root_folder+"/dataset_interference.h5"

def read_hdf5(output_path):

    f =  h5.File(output_path,'r')
    m = f['iq']
    l = f['labels']
    s = f['snrs']
    #print(m.shape)

    return m,l,s


def shuffle(*datas):

    for d in datas:
        random.seed(444)
        random.shuffle(d)



if __name__=="__main__":
    a = list(range(6))
    b = list(range(6))
    c = list(range(6))
    shuffle(a,b,c)
    print(a,b,c)