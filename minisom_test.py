from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(False)
from sklearn.preprocessing import LabelEncoder
from colors import *
import matplotlib.gridspec as gridspec
from dataloader import label_idx
import h5py as h5


def som_plot(som, data, data_name, labels, title=False):
    # CREATE OBJECTS
    fig = plt.figure(figsize=(7, 7), facecolor='white')
    gs1 = gridspec.GridSpec(11, 5)
    ax1 = plt.subplot(gs1[:10, :])
    ax3 = plt.subplot(gs1[-1:, :])

    colors = iter(color_array_solid)
    distance_map = som.distance_map()
    som_size = distance_map.shape
    pc = ax1.pcolor(distance_map.T, cmap="Greys")
    ax1.set_aspect('equal')

    if (title):
        # ax1.set_title(title)
        fig.text(0.72, 0.35, title.replace(" | ", "\n"), fontsize=10)

    # get all winners
    num_points = 512
    w = np.zeros((num_points, 2))
    for i in range(num_points):
        w[i:] = som.winner(data[i])
    # print(w.shape)

    le = LabelEncoder()
    mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                        "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
    le.fit(mod_schemes)
    labels=labels[:num_points]

    for mod in le.transform(mod_schemes):
        # print(np.where(labels == i))
        w_i = w[np.where(labels == mod)]
        ax1.scatter(w_i[:,0] + .5, w_i[:,1] + .5, color=next(colors), alpha=0.5, label=mod_schemes[mod])

    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_ticks([])
        pass

    # Legend & Colorbar
    cb = fig.colorbar(pc, cax=ax3, orientation="horizontal")
    cb.set_ticks([])
    ax3.set_xlabel("Distance")

    legnd = ax1.legend(bbox_to_anchor=(1.0, 1.02), loc="upper left")
    for handle in legnd.legendHandles:
        handle.set_alpha(1)

    fig.subplots_adjust(left=0.05, right=0.7)
    plt.savefig("minisom_test_6.png")
    plt.show()


def train_som(path):

    data = np.load(path,allow_pickle=True)
    iq = data['matrix']
    labels = data['labels']

    train_bound = int(0.75 * labels.shape[0])
    val_bound = int(0.80 * labels.shape[0])

    x_train = np.array(iq[:train_bound])
    y_train = np.array(labels[:train_bound])

    data_=list()
    for _, x in enumerate(x_train):
        # print(x)
        y = x.flatten()
        data_.append(y)

    # print(np.array(data_).shape)  # N x features

    mods=[]
    for label in y_train:
        mods.append(label_idx(label))

    som = MiniSom(16, 16, 256, sigma=0.7, learning_rate=0.2)  # initialization of SOM
    som.random_weights_init(data_)
    # print(som.get_weights())
    som.train_random(data_,10000)  # trains the SOM
    # print(som.get_weights())
    print("model trained")
    som_plot(som,data_,data_name=None,labels=mods)


def train_with_cnn_features(path):
    f = h5.File(path, 'r')
    features,t_labels,pred_labels = f['features'],f['true_labels'],f['pred_labels']


    som = MiniSom(16, 16, 128, sigma=0.7, learning_rate=0.2)  # initialization of SOM
    som.random_weights_init(features)
    # print(som.get_weights())
    som.train_random(features, 10000)  # trains the SOM
    # print(som.get_weights())
    print("model trained")
    som_plot(som, features, data_name=None, labels=pred_labels)


if __name__ == "__main__":
    # train_som("/media/backup/Arsenal/rf_dataset_inets/test_sample.npz")
    train_with_cnn_features("/media/backup/Arsenal/rf_dataset_inets/feature_set_vsg20.h5")