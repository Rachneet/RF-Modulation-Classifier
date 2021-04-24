import numpy as np


def read_filtered(file_dir):
    data = np.load(file_dir)
    matrix = data['matrix']
    snrs = data['snrs']
    labels = data['labels']
    return (labels)


def sort_matrix_entries(matrix):
    sorted = [[matrix[i].real,matrix[i].imag] for i in range(matrix.shape[0])]
    return sorted

# one hot encoding the labels
def encode_labels(num_categories,labels):
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_categories)[targets]

    return one_hot_targets