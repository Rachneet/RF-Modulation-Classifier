import numpy as np
from Receiver import *
from dataloader import label_idx
from cnn_model import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from PIL import Image
import librosa
import librosa.display
import h5py as h5
import os
import cv2
import gc
import csv
from sklearn import preprocessing
from tqdm import tqdm
import scipy.signal as sig
import read_h5 as reader
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def visualize_signal(iq_signal):

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
    dummyReceiver.visualize_samples(mode='filtered',filtered_data=np.array(iq_signal))


class SaveFeatures():
    def __init__(self, module, backward=False):
        if backward==False:
            if isinstance(module, nn.Sequential):
                for name,layer in module._modules.items():
                    self.hook = layer.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_forward_hook(self.hook_fn)
        else:
            if isinstance(module, nn.Sequential):
                for name,layer in module._modules.items():
                    # print(name,layer)
                    self.hook = layer.register_backward_hook(self.hook_fn_backward)
            else:
                self.hook = module.register_backward_hook(self.hook_fn_backward)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()

    def hook_fn_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out

    def close(self):
        self.hook.remove()


def save_cnn_features():
    # data = np.load("/media/backup/Arsenal/rf_dataset_inets/dataset_interpretation.npz", allow_pickle=True)
    # iq = data['matrix']
    # labels = data['labels']
    path_h5 = "/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5"
    iq, labels, snrs = reader.read_hdf5(path_h5)
    df = pd.DataFrame()
    df['iq'] = list(map(lambda x: np.array(x, dtype=np.float32), iq))
    df['normalized_iq'] = df.iq.apply(lambda x: preprocessing.scale(x, with_mean=False))
    df.drop('iq', axis=1, inplace=True)
    df['labels'] = list(map(lambda x: np.array(x, dtype=np.float32), labels))
    df = df.sample(frac=1, random_state=4)
    train_bound = int(0.75 * df.labels.shape[0])

    x_train = DataLoader(df['normalized_iq'][:train_bound].values,batch_size=512,shuffle=False)
    y_train = DataLoader(df['labels'][:train_bound].values,batch_size=512,shuffle=False)

    model = torch.load("trained_cnn_intf_free_vsg20",map_location='cuda:0')
    model.eval()
    activations = SaveFeatures(list(model.children())[7])

    output_path = "/media/backup/Arsenal/rf_dataset_inets/feature_set_training_fc8_vsg20.h5"
    num_features=8

    for iter,batch in enumerate(zip(x_train,y_train)):
        act_features = []
        test_true = []
        test_prob = []
        _, n_true_label = batch

        batch = [Variable(record).cuda() for record in batch]

        t_data, _ = batch
        t_predicted_label = model(t_data)
        features = t_predicted_label.detach().cpu().data.numpy()
        # features = activations.features.detach().cpu().data.numpy()
        features = features.squeeze()
        # features = features.reshape(*features.shape[:1],-1)
        # print(features.shape)
        # print(features)

        act_features.append(features)
        test_prob.append(t_predicted_label)
        test_true.extend(n_true_label.cpu().data.numpy())
        test_prob = torch.cat(test_prob, 0)
        test_prob = test_prob.cpu().data.numpy()
        y_pred = np.array(np.argmax(test_prob, -1))
        y_true = np.array(np.argmax(test_true, -1))
        act_features = np.array(act_features).squeeze()
        print(np.array(act_features).shape)
        # print(y_pred)
        # print(y_true)

        if not os.path.exists(output_path):
            with h5.File(output_path,'w') as hdf:
                hdf.create_dataset('features',data=act_features,chunks=True,maxshape=(None,num_features),compression='gzip')
                hdf.create_dataset('pred_labels', data=y_pred, chunks=True, maxshape=(None,), compression='gzip')
                hdf.create_dataset('true_labels', data=y_true, chunks=True, maxshape=(None,), compression='gzip')
                print(hdf['features'].shape)

        else:
            with h5.File(output_path, 'a') as hf:
                hf["features"].resize((hf["features"].shape[0] + act_features.shape[0]), axis=0)
                hf["features"][-act_features.shape[0]:] = act_features
                hf["pred_labels"].resize((hf["pred_labels"].shape[0] + y_pred.shape[0]), axis=0)
                hf["pred_labels"][-y_pred.shape[0]:] = y_pred
                hf["true_labels"].resize((hf["true_labels"].shape[0] + y_true.shape[0]), axis=0)
                hf["true_labels"][-y_true.shape[0]:] = y_true
                print(hf['features'].shape)



def plot_activations(iq_sample):
    data = np.load("/media/backup/Arsenal/rf_dataset_inets/dataset_interpretation.npz", allow_pickle=True)
    iq = data['matrix']
    labels = data['labels']

    input = np.array(iq_sample)
    # print(input.dtype)
    # visualize_signal(complex)
    input = torch.Tensor(input).cuda()
    input = input.unsqueeze(dim=0)  # adding batch dimension

    # pass to pre trained model
    model = torch.load("trained_cnn_intf_free_vsg20")
    model.eval()

    # get activations for 6th convolution layer
    activations = SaveFeatures(list(model.children())[5])

    output = model(input)
    output = output.detach().cpu().data.numpy()
    features = activations.features.detach().cpu().data.numpy()
    # print(features.shape)
    features = features.flatten()
    # print(np.argmax(features))
    features = np.reshape(features, (-1, 8, 8))
    features = features.transpose(1, 2, 0)

    # img = Image.fromarray(features, 'L')

    # plt.imshow(features[:, :, 0])
    # plt.show()

# convert iq samples to complex form
def iq_to_complex(iq_sample):
    complex = []
    for sample in iq_sample:
        re = sample[0]
        img = sample[1]
        signal = np.complex(re + 1j * img)
        complex.append(signal)
    return complex


def complex_to_iq(complex):
    iq = []
    signal=[]
    for sample in complex:
        re = sample.real
        img = sample.imag
        iq.append(np.array([re,img]))
    return np.array(iq)


# basically grad-cam
def visualizing_filters(iq_sample):

    # pass to pre trained model
    model = torch.load("trained_cnn_intf_free_vsg20")
    model.eval()


    # get activations for 6th convolution layer
    activations = SaveFeatures(list(model.children())[5])
    grads = SaveFeatures(list(model.children())[5], backward=True)

    pred = model(torch.Tensor(iq_sample).unsqueeze(dim=0).cuda())
    idx = pred.argmax(dim=1).item()
    print(idx)
    pred[:,idx].backward()
    features = activations.features
    # print(grads.gradients[0].size())
    pooled_gradients = torch.mean(grads.gradients[0], dim=[0, 2, 3])  # global avg pooled over the ht and wt dims
                                                                     # to obatin the neuron importance wts.
    # weight the channels by corresponding gradients
    # weighted combination of forward activation maps
    for i in range(64):
        features[:, i, :, :] *= pooled_gradients[i]

    # Note: this has to be followed by Relu acc to the paper
    # consider adding it
    # this will give us a heatmap of same dims as the last conv layer dims; 14x14 in case of VGG for example

    features = features.detach().cpu().data.numpy()
    features = features.flatten()
    features = np.reshape(features, (-1, 8, 8))
    features = features.transpose(1, 2, 0)

    # plt.imshow(features[:, :, 0])
    # plt.show()

    heatmap = features[:, :, 0]
    img = cv2.imread('signal_test.png')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    # print(superimposed_img.shape)
    plt.imshow(superimposed_img[:,:,0])
    plt.show()
    cv2.imwrite('heatmap.jpg', superimposed_img)


    # heatmap = torch.mean(features, dim=1).squeeze(dim=2)
    # print(heatmap.size())
    # # normalize the heatmap
    # heatmap /= torch.max(heatmap)
    # heatmap = heatmap.detach().cpu().data.numpy()
    # # draw the heatmap
    # plt.imshow(heatmap)
    # plt.show()

    # heatmap = torch.mean(features, dim=1).squeeze()
    # print(heatmap.size())
    # plt.imshow(heatmap.detach().cpu().data.numpy())
    # plt.show()

    # np.random.seed(123)
    # visualize_signal(iq_to_complex(iq_sample))
    # scaler = preprocessing.StandardScaler(with_mean=False)
    # scaler.fit(iq_sample)
    # rand_input = scaler.transform(iq_sample)
    # print(rand_input)
    # rand_input = torch.Tensor(rand_input).cuda()
    # rand_input = rand_input.unsqueeze(dim=0)
    # rand_input = Variable(rand_input, requires_grad=True).cuda()
    # optimizer = torch.optim.Adam([rand_input], lr=0.01, weight_decay=1e-6)
    #
    # # The network weights are fixed, the network will not be trained, and we
    # # try to find an image(or iq values in our case) that maximizes the average activation of a certain
    # # feature map by performing gradient descent optimization on the pixel values.
    #
    # for n in range(20):
    #     optimizer.zero_grad()
    #     out = model(rand_input)
    #     # print(out)
    #     # print(activations.features)
    #     # print(activations.features[0,0])
    #     loss = -1 * activations.features[0,33].mean()  # taking filter ## as it showed max activation
    #     loss.backward(retain_graph=True)
    #     print(grads.features)
    #     optimizer.step()
    #
    # val = rand_input.data.cpu().numpy()[0]
    # val = scaler.inverse_transform(val)
    # # visualize_signal(iq_to_complex(val))
    # print(val)
    activations.close()


def plot_iq(iq_sample,fig_name):
    # plots a basic iq signal
    fs = 50000000.0
    n_time = int(20 * fs / 10e6)  # show 10 symbols
    # print(n_time)
    i,q=[],[]
    # comp=iq_to_complex(iq_sample)

    for iq in iq_sample:
        i.append(iq[0])
        q.append(iq[1])
    # comp = np.array(comp)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    # f, t, Sxx = sig.spectrogram(comp, fs,return_onesided=False)
    # time_axis=[]
    # for time in t:
    #     time_axis.append(int(time*10e6))

    # ax = plt.figure(num=None, figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
    ax.plot(i[:n_time],color='orangered')
    ax.plot(q[:n_time],color='blue')

    # ax.set_axis_off()
    ax.set_facecolor('xkcd:azure')
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend(loc="upper right")
    plt.savefig(fig_name)

    plt.close()


def generate_image_dset(iq,labels,output_dir,mode=None):

    # output_dir='/media/backup/Arsenal/rf_dataset_inets/image_dataset/'
    # if not os.path.exists(output_dir+"train"):
    #     os.makedirs(output_dir+"train")

    mods = [0,1,2,3,4,5,6,7]
    if  mode=="train":
        for i in tqdm(range(int(0.80*iq.shape[0]))):
            label = label_idx(labels[i])
            if label in mods:
                path = output_dir+"train/"+str(label)
                if not os.path.exists(path):
                    os.makedirs(path)
                plot_iq(iq[i], path + "/Fig_" + str(i) + "_mod_" + str(label_idx(labels[i])) + ".png")
                # if len([name for name in os.listdir(path)])==70000:
                #     mods.remove(label)
                if i==70000:
                    break
            else:
                continue
    elif mode=="test":
        for i in tqdm(range(int(0.80*iq.shape[0]),int(0.81*iq.shape[0]))):
            label = label_idx(labels[i])
            if label in mods:
                path = output_dir+"test/"+str(label)
                if not os.path.exists(path):
                    os.makedirs(path)
                plot_iq(iq[i], path + "/Fig_" + str(i) + "_mod_" + str(label_idx(labels[i])) + ".png")
                # if len([name for name in os.listdir(path)])==250:
                #     mods.remove(label)
                # if len(mods)==0:
                #     break
            else:
                continue


def plot_melspectrogram(iq,fig_name):
    n_fft=256  # frequency spectrum with n_fft bins
    hop_length=128
    sample_rate = 50e6
    fmax=50e6
    fmin=10e4
    iq = iq_to_complex(iq)
    real = [sample.real for sample in iq]
    # iq = iq[:0+n_fft]
    # X = fft.fft(iq)
    # X_magnitude, X_phase = librosa.magphase(X)
    # X_magnitude_db = librosa.amplitude_to_db(X_magnitude)
    # print(X_magnitude)

    # plotting spectrogram
    # stft = librosa.stft(np.array(real), n_fft=n_fft, hop_length=hop_length)
    # stft_magnitude, stft_phase = librosa.magphase(stft)
    # stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
    fig = plt.Figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    mel_spec = librosa.feature.melspectrogram(np.array(real), n_fft=n_fft, hop_length=hop_length,
                                              n_mels=16,
                                              sr=sample_rate,
                                              power=1.0, fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db,x_axis='time',y_axis='mel')
    # librosa.display.specshow(stft_magnitude_db)
    # # plt.specgram(stft_magnitude_db)
    # plt.show()

    plt.savefig(fig_name)
    # plt.clf()
    # plt.show()
    # fig.clear()
    plt.close()
    # gc.collect()


def deep_dream(iq_sample):
    # preprocess iq input
    # pure = np.zeros((1024, 2),dtype=float)
    noise = np.random.normal(0, 1, [1024,2])
    signal = iq_sample+noise
    # print(signal)

    # pass to pre trained model
    model = torch.load("trained_cnn_intf_free_vsg20")
    model.eval()

    # get activations for 6th convolution layer
    activations = SaveFeatures(list(model.children())[5])

    input = torch.Tensor(signal).cuda()
    input = input.unsqueeze(dim=0)
    input = input.permute(0,2,1)
    input = input.unsqueeze(dim=3)
    input = Variable(input, requires_grad=True).cuda()
    optimizer = torch.optim.Adam([input], lr=0.2, weight_decay=1e-6)

    # The network weights are fixed, the network will not be trained, and we
    # try to find an image(or iq values in our case) that maximizes the average activation of a certain
    # feature map by performing gradient descent optimization on the pixel values.


    for i in range(100):
        optimizer.zero_grad()
        x = input
        for index, layer in enumerate(model._modules.items()):
            # print(layer[1])
            # print(index)
            x = layer[1](x)

            if index==5:
                break
        # plot_iq(activations.features.squeeze(),'signal_test.png')
        # print(activations.features.shape)
        # print(activations.features[0,31])
        # print(activations.features[0, 25])
        # print(torch.argmax(activations.features.squeeze()))
        # print(torch.argmax(activations.features,dim=1)[0][0].item())
        # print(torch.argmax(activations.features, dim=2)[0][0].item())
        loss = -torch.mean(activations.features[0,
                torch.argmax(activations.features,dim=1)[0][0].item()])
        print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.detach().cpu().data.numpy()))

        # Backward
        loss.backward()

        # Update signal
        optimizer.step()

        # print(input.shape)
        y = input.squeeze().permute(1,0)
        recreated_signal = y.detach().cpu().data.numpy()-noise
        # print(recreated_signal.shape)
        if i>1 and i%20==0:
            plot_iq(recreated_signal,'signal'+str(i)+".png")

    activations.close()




if __name__=="__main__":

    # dataloader.load_batch("/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5")
    # data = np.load("/media/backup/Arsenal/rf_dataset_inets/dataset_interpretation.npz", allow_pickle=True)
    # iq = data['matrix']
    # labels = data['labels']

    # deep_dream(iq[0])
    # print(iq.shape)
    # print("Modulation: {}".format(labels[:35]))  # bpsk
    # print(np.array(iq[0]))
    # print(label_idx(labels[0]))
    # plot_activations(iq[13])
    # plot_iq(iq[32],fig_name='signal_test.png')
    # visualizing_filters(iq[32])

    # save_cnn_features()
    # path="/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5"
    # iq, labels, snrs = reader.read_hdf5(path)
    # # reader.shuffle(iq,labels,snrs)
    #
    # # analyze data
    # df = pd.DataFrame()
    # df['iq'] = list(map(lambda x: x, iq))
    # df['normalized_iq'] = df.iq.apply(lambda x: preprocessing.scale(x, with_mean=False))
    # df.drop('iq', axis=1, inplace=True)
    # df['labels'] = list(map(lambda x: x, labels))
    # df = df.sample(frac=1, random_state=4)
    #
    # output_dir = '/media/backup/Arsenal/rf_dataset_inets/fig_dataset/'
    # generate_image_dset(df['normalized_iq'].values,df['labels'].values,output_dir,"test ")

    # plot_melspectrogram(iq[13],'test.png')
    pass