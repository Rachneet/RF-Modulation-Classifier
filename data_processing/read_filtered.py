import numpy as np
from sklearn.preprocessing import OneHotEncoder

def read_filtered(file_dir):
    data = np.load(file_dir)
    matrix = data['matrix']
    snrs = data['snrs']
    labels = data['labels']

    # print(matrix.shape, snrs.shape, labels.shape)
    # # print(matrix)
    # print(snrs)
    # print(labels)
    # print(matrix[0])
    # print(matrix[0].shape[0])
    # print(type(matrix[0]))
    return (labels)


def sort_matrix_entries(matrix):
    # l=[]
    # for i in range(matrix.shape[0]):
    #     l.append([matrix[i].real,matrix[i].imag])

    sorted = [[matrix[i].real,matrix[i].imag] for i in range(matrix.shape[0])]

    # print(len(l))
    return sorted

# one hot encoding the labels
def encode_labels(num_categories,labels):
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_categories)[targets]

    return one_hot_targets




if __name__=="__main__":
    file_dir = "/data/all_RFSignal_cdv/interference/ota/no_cfo/tx_usrp/rx_usrp/intf_vsg/i_OFDM_64QAM/10db/sir_10db/npz/OFDM_QPSK_filtered.npz"
    labels = read_filtered(file_dir)
    print(labels)
    # new_arr = np.apply_along_axis(sort_matrix_entries, axis=1, arr=matrix)
    # print(new_arr)
    # print(new_arr.shape)
    print(encode_labels(8, labels))