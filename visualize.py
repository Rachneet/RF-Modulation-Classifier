import numpy as np
from Receiver import *
from dataloader import *
from cnn_model import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image
# import librosa
# import librosa.display
import h5py as h5
import os
import cv2
import gc
import csv
from ast import literal_eval
from sklearn import preprocessing
from tqdm import tqdm
import scipy.signal as sig
import read_h5 as reader
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# plotting libraries
import plotly
import commpy
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
plotly.io.orca.config.save()
pio.renderers.default = 'svg'


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
    path_h5 = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
    training_params = {'batch_size': 512, 'num_workers': 10}
    train_set, val_set, test_set = load_data(path_h5,0.05,0.2,**training_params)

    model = torch.load("/home/rachneet/thesis_results/trained_cnn_no_intf_vsg_all",map_location='cuda:0')
    model.eval()
    # print(list(model.children())[5])
    activations = SaveFeatures(list(model.children())[7])   # using last conv layer
    # print(list(model.children())[5])
    output_path = "/home/rachneet/rf_dataset_inets/feature_set_training_fc8_vsg_all.h5"
    # num_features=8

    for batch in tqdm(train_set):
        print('In loop')
        print(batch)
        act_features = []
        test_true = []
        test_prob = []
        _, n_true_label,_ = batch

        batch = [Variable(record).cuda() for record in batch]

        t_data, _,_ = batch
        t_predicted_label = model(t_data)
        # features = t_predicted_label.detach().cpu().data.numpy()
        features = activations.features.detach().cpu().data.numpy()
        features = features.squeeze()
        features = features.reshape(*features.shape[:1],-1)
        print(features.shape)
        print(features)

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
        break

        # if not os.path.exists(output_path):
        #     with h5.File(output_path,'w') as hdf:
        #         hdf.create_dataset('features',data=act_features,chunks=True,maxshape=(None,num_features),compression='gzip')
        #         hdf.create_dataset('pred_labels', data=y_pred, chunks=True, maxshape=(None,), compression='gzip')
        #         hdf.create_dataset('true_labels', data=y_true, chunks=True, maxshape=(None,), compression='gzip')
        #         print(hdf['features'].shape)
        #
        # else:
        #     with h5.File(output_path, 'a') as hf:
        #         hf["features"].resize((hf["features"].shape[0] + act_features.shape[0]), axis=0)
        #         hf["features"][-act_features.shape[0]:] = act_features
        #         hf["pred_labels"].resize((hf["pred_labels"].shape[0] + y_pred.shape[0]), axis=0)
        #         hf["pred_labels"][-y_pred.shape[0]:] = y_pred
        #         hf["true_labels"].resize((hf["true_labels"].shape[0] + y_true.shape[0]), axis=0)
        #         hf["true_labels"][-y_true.shape[0]:] = y_true
        #         print(hf['features'].shape)



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


def plot_barchart():
    sirs = ['0 dB','5 dB','10 dB','15 dB','20 dB','0-20 dB']
    # sirs = ['5 dB', '10 dB', '15 dB', '20 dB', '5-20 dB']
    # acc = [0.82,0.86,0.87,0.88,0.89]
    # acc2 = [0.83,0.86,0.86,0.87,0.87]
    # acc3 = [0.71,0.80,0.86,0.87,0.87]
    # acc = [0.72,0.76,0.79,0.80,0.77]  # cnn intf
    # acc2 = [0.75,0.80,0.80,0.81,0.79]
    # acc3 = [0.69,0.78,0.80,0.82,0.77]
    # acc = [0.35,0.54,0.68,0.69,0.57]  # cnn intf generalize
    # acc2 = [0.37,0.61,0.75,0.77,0.63]
    # acc3 = [0.38,0.65,0.73,0.75,0.63]
    # acc = [0.60,0.83,0.94,0.99,0.99,0.87]  # cnn cfo generalized
    # acc2 = [0.59,0.83,0.94,0.99,0.99,0.87]
    # acc3 = [0.57,0.80,0.93,0.98,0.99,0.85]
    # acc = [0.89, 0.96, 0.97, 0.99, 1.0, 0.96]
    # acc2 = [0.61, 0.83, 0.94, 0.99, 0.99, 0.87]
    # acc3 = [0.75, 0.98, 1.00, 1.00, 1.00, 0.95]
    # vsg usrp tl
    # acc = [0.61, 0.83, 0.94, 0.99, 0.99, 0.87]   # tl vsg
    # acc2 = [0.51, 0.55, 0.57, 0.49, 0.42, 0.51]
    # acc3 = [0.55, 0.74, 0.85, 0.91, 0.93, 0.80]
    # intf tl
    acc = [0.72,0.76,0.79,0.80,0.77]
    acc2 = [0.35, 0.54, 0.68, 0.69, 0.57]
    acc3 = [0.62,0.74,0.77,0.78,0.73]


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        name="Train: SC BPSK <br>Test: SC BPSK",#'SC_BPSK',
        text=acc,
        textfont=dict(color='black', size=10),
        textposition='auto',
        textangle=90,
        # marker_color='#ff6555',
        # marker_line_color='#d62728'
        marker_color='rgb(128,125,186)',  # purple
        marker_line_color='rgb(84,39,143)'
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        name="Train: None <br>Test: SC BPSK",#'SC_16QAM',
        text=acc2,
        textfont=dict(color='black', size=10),
        textposition='auto',
        textangle=90,
        # marker_color='#66c56c',
        # marker_line_color='#2ca02c'
        marker_color='#23aaff', # blue
        marker_line_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc3,
        name="Train: None <br>Transfer: SC BPSK",#'OFDM_64QAM',
        text=acc3,
        textfont=dict(color='black', size=10),
        textposition='auto',
        textangle=90,
        # marker_color='#f4b247',
        # marker_line_color='#ff7f0e'
        marker_color='rgb(253,141,60)',  # orange
        marker_line_color='rgb(241,105,19)'
    ))

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='inside',
                     mirror=True,
                     linecolor='black',
                     linewidth=1,
                     tickmode='linear',
                     tick0=0,
                     dtick=0.1,
                     range=[0,1.09],
                     tickfont=dict(family="times new roman", size=15, color='black'),
                     title=dict(
                         font=dict(
                             family="times new roman",
                             size=14,
                             color="black"
                         ),
                     text='Classification Accuracy')
                     )
    fig.update_xaxes(automargin=True, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=1,
                     tickfont=dict(family="times new roman", size=15, color='black'),
                     title=dict(
                         font=dict(
                             family="times new roman",
                             size=14,
                             color="black"
                         ),
                     text='SIR'
                     )
                     )
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=1, opacity=0.8)

    fig.update_layout(
        # title_text='<b>CNN Performance with Interfering Signals <br>at SNR 10 dB',
        # title_x=0.50,
        # title_y=0.95,
        margin=dict(b=350, l=0, r=170, t=20),
        yaxis={"mirror": "all"},
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.1,
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            # x=0.25,
            x=0.1,
            y=1.2,
            font=dict(
                # family="sans-serif",
                size=12,
                color="black"
            ),
            traceorder='normal'
        )
    )

    plotly.offline.plot(fig, filename='test_bar.html', image='svg')


def plot_heatmap():
    res = [[0.0,0.688,	0.746,	0.772,	0.780],
        [0.592,	0.0,0.865,	0.892,	0.900],
        [0.549,	0.792,	0.0,	0.970,	0.981],
        [0.508,	0.724,	0.894,	0.0,0.993],
        [0.466,	0.658,	0.840,	0.959,	0.0]]

    x = ['0 dB','5 dB','10 dB','15 dB','20 dB']
    z_text = [[str(y) for y in x] for x in np.array(res[::-1])]
    temp2=[]
    for row in z_text:
        temp=[]
        for item in row:
            if item == '0.0':
                val = item.replace('0.0','-')
                temp.append(val)
            else:
                temp.append(item)
        temp2.append(temp)
    # print(temp2)

    # set up figure
    # colorscale = [[0, 'white'], [1, 'rgb(115,154,228)']]
    colorscale = [[0.0, 'rgb(255,255,255)'], [.2, 'rgb(255, 255, 153)'],
                  [.4, 'rgb(153, 255, 204)'], [.6, 'rgb(179, 217, 255)'],
                  [.8, 'rgb(240, 179, 255)'], [1.0, 'rgb(255, 77, 148)']]
    font_colors = ['black']
    fig = ff.create_annotated_heatmap(list(reversed(res)), x=x, y=list(reversed(x)), annotation_text=temp2,
                                      colorscale=colorscale, font_colors=font_colors,zmin=0,zmax=1)
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig['data'][0].colorbar = dict( #title='Accuracy <br> ',
                                   outlinecolor="black",
                                   outlinewidth=1,
                                   ticks='outside',
                                   tickmode='linear',
                                    tickfont=dict(
                                        family="times new roman",
                                        size=14,
                                        color="black"
                                    ),
                                   tick0=0,
                                   dtick=0.1,
                                   len=1.08
                                   )
    # fig.update_layout(title_text='<b>CNN Performance <br>with Transfer Learning Approach',
    #                   # xaxis = dict(title='x'),
    #                   # yaxis = dict(title='x')
    #                   )
    # add custom colorbar title
    # fig.add_annotation(dict(font=dict(color="black"), #size=14),
    #                         x=1.13,
    #                         y=1.07,
    #                         showarrow=False,
    #                         text="Accuracy",
    #                         xref="paper",
    #                         yref="paper"))

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     color='black',
                     tickfont=dict(
                         family="times new roman",
                         size=14,
                         color="black"
                     ),
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=18,
                             color="black"
                         )),
                     )
    fig.update_xaxes(automargin=True,
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     color='black',
                     tickfont=dict(
                         family="times new roman",
                         size=14,
                         color="black"
                     ),
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=18,
                             color="black",
                         )),
                     )
    # # add custom xaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=0.5,
    #                         y=1.15,
    #                         showarrow=False,
    #                         text="SNR (Test)",
    #                         xref="paper",
    #                         yref="paper"))
    # # add custom yaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=-0.15,
    #                         y=0.5,
    #                         showarrow=False,
    #                         textangle=-90,
    #                         text="SNR (Train)",
    #                         xref="paper",
    #                         yref="paper"))
    fig.update_layout(
        margin=dict(b=300, l=150, r=350,t=20),
        title_x=0.50,
        title_y=0.1,
        paper_bgcolor="white",
        plot_bgcolor='white',
        # check if legend needed
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            x=0,
            y=1,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        )
    )
    # Make text size smaller
    # for i in range(len(fig.layout.annotations)):
    #     fig.layout.annotations[i].font.size = 12

    plotly.offline.plot(fig, filename="test_heatmap.html", image='svg')


def draw_comparison_chart():

    sirs = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB', '0-20 dB']
    acc = [0.58, 0.80, 0.93, 0.98, 0.99, 0.86]
    # acc2 = [0.57,0.79,0.91,0.98,0.99,0.85]  # usrp
    acc2 = [0.59, 0.81, 0.94, 0.99, 1.0, 0.87]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        # name="             ",
        name="              ",  # 'CNN (9-Layer)',
        text=acc,
        textfont=dict(color='black', size=10),
        textposition='auto',
        # marker_color='#66c56c',  # green
        # marker_line_color='#2ca02c',
        # marker_color='#ff6555', # red
        # marker_line_color='#d62728',
        # marker_color='rgb(128,125,186)',  # purple
        # marker_line_color='rgb(84,39,143)',
        marker_color='#23aaff', # blue
        marker_line_color='#1f77b4',
        # marker_color='rgb(103,219,165)',  # mint
        # marker_line_color='rgb(76,200,163)',
        # width=0.35,
        # offset=-0.32
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        # name="",  #'ResNet-101',
        name="                                ",
        text=acc2,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#f4b247',  # yellow
        marker_line_color='#ff7f0e',
        # marker_color='#66c56c',  # green
        # marker_line_color='#2ca02c',
        # marker_color='#23aaff', # blue
        # marker_line_color='#1f77b4',
        # marker_color='#ff6555', # red
        # marker_line_color='#d62728',
        # marker_color='rgb(193,118,111)',  # brown
        # marker_line_color='rgb(166,84,97)',
        # width=0.35,
        # offset=0.02
    ))


    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='inside',
                     mirror=True,
                     tickfont=dict(family="times new roman",size=15,color='black'),
                     linecolor='black',
                     linewidth=1,
                     tickmode='linear',
                     tick0=0,
                     dtick=0.1,
                     range=[0, 1.05],
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         # text='Accuracy')
                     ))
    fig.update_xaxes(automargin=True, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     tickfont=dict(family="times new roman",size=15,color='black'),
                     linecolor='black',
                     linewidth=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         # text='SIR'
                     )
                     )
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=1, opacity=0.8)

    fig.update_layout(
        # title_text='<b>Model Comparison with Interfering OFDM Signals <br>at SNR 10 dB',
        # margin=dict(b=260, l=0, r=150, t=20), # for small boxes
        margin=dict(b=350, l=0, r=200, t=20),  # normal
        title_x=0.50,
        title_y=0.90,
        yaxis={"mirror": "all"},
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.1,
        # width=500,
        # height=500,
        # showlegend=False
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            # x=0.4,  # for hw comp
            x=0.25, #for model comp
            y=1.2,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        )
    )
# image_width=400, image_height=400

    plotly.offline.plot(figure_or_data=fig, filename='test_bar.html', image='svg', image_height=550, image_width=600)


def plt_tl_chart():
    sirs = ['0 dB','5 dB', '10 dB', '15 dB', '20 dB', '0-20 dB']
    # acc = [0.82,0.86,0.87,0.88,0.89]
    # acc2 = [0.83,0.86,0.86,0.87,0.87]
    # acc3 = [0.71,0.80,0.86,0.87,0.87]
    # intf tl cases in order
    # acc = [0.72,0.76,0.79,0.80,0.77]
    # acc2 = [0.62,0.74,0.77,0.78,0.73]
    # acc = [0.75,0.80,0.80,0.81,0.79]
    # acc2 = [0.71,0.80,0.82,0.82,0.79]
    # acc = [0.69,0.78,0.80,0.82,0.77]
    # acc2 = [0.66,0.76,0.79,0.80,0.75]
    # vsg usrp tl
    # acc = [0.61, 0.83, 0.94, 0.99, 0.99, 0.87]   # tl vsg
    # acc2 = [0.55, 0.74, 0.85, 0.91, 0.93, 0.80]
    # acc=[0.57,0.79,0.91,0.98,0.99,0.85]          # tl usrp
    # acc2 = [0.56, 0.74, 0.86, 0.93, 0.94, 0.81]
    # vsg cfo1
    # acc = [0.58, 0.80, 0.93, 0.98, 0.99, 0.86]
    # acc2 = [0.75, 0.98, 1.0, 1.0, 1.0, 0.95]
    # vsg cfo5
    # acc = [0.59, 0.81, 0.94, 0.99, 1.0, 0.87]
    # acc2 = [0.71, 0.97, 1.0, 1.0, 1.0, 0.94]
    acc = [0.69, 0.82, 0.92, 1.0, 1.0, 0.88]
    acc2 = [0.62, 0.79, 0.83, 0.84, 0.85, 0.79]




    fig = go.Figure()
    # fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]], print_grid=True)
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        # name="                  ",#'Traditional <br>Learning',
        name = 'Traditional <br>Learning',
        # text=acc,
        # textfont=dict(color='black', size=10),
        textposition='auto',
        # marker_color='#66c56c',  # green
        # marker_line_color='#2ca02c',
        # marker_color='rgb(103,219,165)',  # mint
        # marker_line_color='rgb(76,200,163)',
        marker_color='#ff6555', # red
        marker_line_color='#d62728',
        # width=0.25,
        # offset=-0.32
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        # name="                  ",#'Transfer <br>Learning (TL)',
        name = 'Transfer <br>Learning (TL)',
        # text=acc2,
        # textfont=dict(color='black', size=10),
        # textposition='auto',
        # marker_color='#f4b247',
        # marker_line_color='#ff7f0e'
        # marker_color='rgb(193,118,111)',  # brown
        # marker_line_color='rgb(166,84,97)',
        marker_color='#66c56c',
        marker_line_color='#2ca02c',
        # width=0.25,
        # offset=0.02
    ))

    # fig.add_shape(type="line", x0=-0.5, y0=3.13, x1=5.5, y1=3.13,    # x1=4.5
    #               line=dict(
    #                   color="#23aaff",
    #                   width=4,
    #                   dash="dash",
    #               ),
    #               yref='y2',
    #               # name='Training time <br>with TL'
    #               # name="                  ",
    #
    #               )
    #
    # fig.add_shape(type="line", x0=-0.5, y0=25.3, x1=5.5, y1=25.3,
    #               line=dict(
    #                   color="#ff6555",
    #                   width=4,
    #                   dash="dash",
    #               ),
    #               yref='y2',
    #               # name='Training time <br>w/o TL'
    #               # name="                  ",
    #
    #               )

    # fig.add_trace(go.Scatter(
    #     x=sirs,
    #     y=[100]*5,
    #     # name='Training time <br>with TL',
    #     name="              ",
    #     mode='lines',
    #     line=dict(
    #         color="#23aaff",
    #         width=12,
    #         dash="dash",
    #     ),
    #     marker_color='rgba(5, 112, 176, .8)',
    #     marker_line_width=8,
    #     yaxis='y2',
    #
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=sirs,
    #     y=[100] * 5,
    #     # name='Training time <br>w/o TL',
    #     name="              ",
    #     mode='lines',
    #     line=dict(
    #         color="#ff6555",
    #         width=12,
    #         dash="dash",
    #     ),
    #     marker_color='rgba(152, 0, 0, .8)',
    #     marker_line_width=8,
    #     yaxis='y2'
    # ))

    fig.update_yaxes(automargin=True,showline=True,ticks='inside',mirror=True,
                     linecolor='black',linewidth=1,tickmode='linear',tick0=0,
                     tickfont=dict(
                         family="times new roman",
                         size=14,
                         color="black"
                     ),
                     dtick=0.1,range=[0, 1.05],title=dict(font=dict(family="times new roman",
                             size=14,
                             color="black"
                         ),
                         text='Accuracy'
                     ))

    fig.update_xaxes(automargin=True, side='bottom',showline=True,ticks='outside',
                     mirror=True,linecolor='black',linewidth=1,
                     tickfont=dict(
                         family="times new roman",
                         size=14,
                         color="black"
                     ),
                     title=dict(
                         font=dict(
                             family="times new roman",
                             size=14,
                             color="black"
                         ),
                         text='SNR'
                     ))
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=2, opacity=0.8)

    fig.update_layout(
        # title_text='<b>Comparison of Learning Approaches <br>with Interfering OFDM Signals at SNR 10 dB',
        # title_x=0.50,
        # title_y=0.90,
        yaxis={"mirror": "all"},
        margin=dict(b=220, l=0, r=260, t=20),
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.15,
        width=500,
        height=500,
        # showlegend=False,
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            x=0,
            y=1.2,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        ),
        yaxis2=dict(
        # title='Training Time (Hours)',
        # title_font=dict(color='black'),
        side='right',
        tickmode='linear',
        tickfont=dict(family='times new roman',size=22,color='black'),
        tick0=0,
        ticks='inside',
        dtick=5,
        range=[0,52.5],
        overlaying= 'y'
    ),
    )

    plotly.offline.plot(figure_or_data=fig, image_width=610, image_height=500, filename='test_bar.html', image='svg')


def train_fraction_effect():
    snrs = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB']
    frac15 = [0.55, 0.74, 0.85, 0.91, 0.93]
    frac30 = [0.59, 0.8, 0.91, 0.97, 0.99]
    frac45 = [0.59, 0.81, 0.92, 0.98, 0.99]
    frac60 = [0.6, 0.82, 0.93, 0.98, 0.99]
    frac75 = [0.61, 0.83, 0.94, 0.99, 0.99]
    y_dict = {'15': frac15, '30': frac30, '45': frac45, '60': frac60, '75': frac75}

    fig = go.Figure()
    # fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]], print_grid=True)
    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    names = ['15','30','45','60', '75']
    fracs = ['15%','30%','45%','60%', '75%']
    colors = ['rgba(255, 101, 85, 1)','rgba(35, 170, 255, 1)','rgba(244, 178, 71, 1)','rgba(102, 197, 108, 1)',
              'rgb(159,130,206)']

    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=snrs,y=y_dict[names[i]],
                                 name="         ",
            # name=fracs[i],
            mode='lines+markers',marker_color=colors[i],
            # marker_color='#f4b247', marker_line_color='#ff7f0e',
            # width=0.25
        ))

    line_ax = [0.5,0.6,0.7,0.8,0.9,1]
    for i in line_ax:
        fig.add_shape(type="line",x0=-0.3,y0=i,x1=4.3,y1=i,
            line=dict(
                color="grey",
                width=1,
                dash="dashdot",
            ),
        )

    fig.update_yaxes(automargin=True, showline=True, ticks="", mirror=True,
                     linecolor='black', linewidth=1,
                     tickfont=dict(color="black",family='times new roman',size=16),
                     title=dict(font=dict( color="black",# family="sans-serif",size=15,
                    ),
                    # text='Classification accuracy'
                    ))

    fig.update_xaxes(automargin=True, side='bottom', showline=True, ticks='inside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(color="black",family='times new roman',size=16),
                     title=dict(font=dict(
                             # family="sans-serif",
                             # size=15,
                             color="black"
                         ),
                         # text='SNR(dB)'
                     ))

    fig.update_traces(marker_line_width=2)
    # fig.add_annotation(dict(font=dict(color="black", size=16,family='times new roman'),
    #                         x=0.95,
    #                         y=0.30,
    #                         showarrow=False,
    #                         # text="Number of IQ samples per signal segment N:",
    #                         xref="paper",
    #                         yref="paper"))
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=110,  # right margin  130
            b=160,  # bottom margin
            t=20,  # top margin
        ),
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        xaxis={"mirror": "all"},
        paper_bgcolor='white',plot_bgcolor='rgba(0,0,0,0)',
        width=500,height=500,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)',orientation='h',
            itemsizing='constant',x=0.22,y=0.3,
            font=dict(size=12, color="black",family='times new roman'),traceorder='normal'
        ),
    )
    plotly.offline.plot(figure_or_data=fig, image_width=600, image_height=500, filename='train_chart.html', image='svg')


def plot_line_chart():
    colors = ['rgba(255, 101, 85, 1)', 'rgba(35, 170, 255, 1)', 'rgba(244, 178, 71, 1)', 'rgba(42, 187, 155, 1)']
    # c_cfo1 = [0.577, 0.801, 0.927, 0.98, 0.992]
    # r_cfo1 = [0.753, 0.983, 1, 1, 1]
    # c_cfo5 = [0.592, 0.814, 0.937, 0.986, 0.995]
    # r_cfo5 = [0.709, 0.966, 0.999, 1, 1]

    c_bpsk = [0.72, 0.76, 0.79, 0.8]
    c_16qam = [0.75, 0.8, 0.8, 0.81]
    c_ofdm = [0.69, 0.78, 0.8, 0.82]
    r_bpsk = [0.86, 0.89, 0.89, 0.89]
    r_16qam = [0.81, 0.85, 0.86, 0.86]
    r_ofdm = [0.78, 0.86, 0.87, 0.87]

    cfo1 = [0.603, 0.829, 0.941, 0.986, 0.995]
    cfo2 = [0.594, 0.828, 0.939, 0.986, 0.994, 0.868]
    cfo3 = [0.569, 0.801, 0.926, 0.980, 0.991, 0.853]
    cfo4 = [0.586, 0.814, 0.936, 0.987, 0.995, 0.863]

    intf1 = [0.35, 0.54, 0.68, 0.69]
    intf2 = [0.37, 0.61, 0.75, 0.77]
    intf3 = [0.38, 0.65, 0.73, 0.75]



    x = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB']
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=cfo1, mode="lines+markers", marker_symbol='circle', marker_color=colors[0],
                             name='     <br>                                              '))
    fig.add_trace(go.Scatter(x=x, y=cfo2, mode="lines+markers", marker_symbol='circle', marker_color=colors[1],
                             name='       <br>                                             '))
    fig.add_trace(go.Scatter(x=x, y=cfo3, mode="lines+markers", marker_symbol='circle', marker_color=colors[2],
                             name='           <br>                          '))
    fig.add_trace(go.Scatter(x=x, y=cfo4, mode="lines+markers", marker_symbol='circle', marker_color=colors[3],
                             name='         <br>                            '))
    # fig.add_trace(go.Scatter(x=x, y=cfo4, mode="lines+markers", marker_symbol='circle', marker_color=colors[3],
    #                          name='                                                    '))
    # fig.add_trace(go.Scatter(x=x, y=r_bpsk, mode="lines+markers", marker_symbol='circle', marker_color=colors[1],
    #                          name='                                                '))
    #
    # fig.add_trace(go.Scatter(x=x, y=c_16qam, mode="lines+markers", marker_symbol='square', marker_color=colors[0],
    #                          name='                                                ',
    #                          line=dict(
    #                              # color="grey",
    #                              # width=1,
    #                              dash="dashdot",
    #                          )
    #                          ))
    # fig.add_trace(go.Scatter(x=x, y=r_16qam, mode="lines+markers", marker_symbol='square', marker_color=colors[1],
    #                          name='                                                 ',
    #                          line=dict(
    #                              # color="grey",
    #                              # width=1,
    #                              dash="dashdot",
    #                          )
    #                          ))
    #
    # fig.add_trace(go.Scatter(x=x, y=c_ofdm, mode="lines+markers", marker_symbol='diamond', marker_color=colors[0],
    #                          name='                                               ',
    #                          line=dict(
    #                              # color="grey",
    #                              # width=1,
    #                              dash="dot",
    #                          )
    #                          ))
    #
    # fig.add_trace(go.Scatter(x=x, y=r_ofdm, mode="lines+markers", marker_symbol='diamond', marker_color=colors[1],
    #                              name='                                                ',
    #                              line=dict(
    #                                  # color="grey",
    #                                  # width=1,
    #                                  dash="dot",
    #                              )
    #                              ))

    line_ax = [0.6, 0.7, 0.8, 0.9, 1]  # change
    for i in line_ax:
        fig.add_shape(type="line", x0=-0.3, y0=i, x1=4.3, y1=i,  # change
                      line=dict(
                          color="grey",
                          width=1,
                          dash="dashdot",
                      ),
                      )

    fig.update_yaxes(automargin=True, showline=True, ticks="", mirror=True,
                     linecolor='black', linewidth=1,
                     tickfont=dict(color="black", family='times new roman', size=16),
                     tick0=0.5,  # change
                     dtick=0.1,
                     range=[0.5, 1.05],
                     title=dict(font=dict(color="black",  # family="sans-serif",size=15,
                                          ),
                                # text='Accuracy')
                                ))

    fig.update_xaxes(automargin=True, side='bottom', showline=True, ticks='inside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(color="black", family='times new roman', size=16),
                     # tick0=0.5,
                     # dtick=0.1,
                     # range=[0.5, 1.05],
                     title=dict(font=dict(
                         # family="sans-serif",
                         # size=15,
                         color="black"
                     ),
                         # text='SNR'
                     ))

    fig.update_traces(marker_line_width=2)
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=110,  # right margin  130
            b=160,  # bottom margin
            t=20,  # top margin
        ),
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        xaxis={"mirror": "all"},
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        width=500, height=500,
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)', orientation='h',
            itemsizing='constant', x=0.05, y=1.5,
            font=dict(size=12, color="black", family='times new roman'), traceorder='normal'
        ),
    )
    plotly.offline.plot(figure_or_data=fig, image_width=600, image_height=400, filename='gen_cfo.html', image='svg')




def scalability_chart():
    snrs = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB']
    acc_256 = [0.474, 0.654, 0.794, 0.888, 0.919]
    acc_512 = [0.536, 0.74, 0.871, 0.952, 0.975]
    acc_1024 = [0.609, 0.83, 0.942, 0.986, 0.995]
    acc_2048 = [0.68, 0.9, 0.983, 0.998, 0.999]
    y_dict = {'256': acc_256, '512': acc_512, '1024': acc_1024, '2048': acc_2048}

    fig = go.Figure()
    # fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]], print_grid=True)
    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    names = ['256','512','1024','2048']
    colors = ['rgba(255, 101, 85, 1)','rgba(35, 170, 255, 1)','rgba(244, 178, 71, 1)','rgba(102, 197, 108, 1)']

    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=snrs,y=y_dict[names[i]],name="         ",#names[i],
            mode='lines+markers',marker_color=colors[i],
            # marker_color='#f4b247', marker_line_color='#ff7f0e',
            # width=0.25
        ))

    line_ax = [0.5,0.6,0.7,0.8,0.9,1]
    for i in line_ax:
        fig.add_shape(type="line",x0=-0.3,y0=i,x1=4.3,y1=i,
            line=dict(
                color="grey",
                width=1,
                dash="dashdot",
            ),
        )

    fig.update_yaxes(automargin=True, showline=True, ticks="", mirror=True,
                     linecolor='black', linewidth=1,
                     tickfont=dict(color="black",family='times new roman',size=16),
                     title=dict(font=dict( color="black",# family="sans-serif",size=15,
                    ),
                    # text='Accuracy')
                    ))

    fig.update_xaxes(automargin=True, side='bottom', showline=True, ticks='inside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(color="black",family='times new roman',size=16),
                     title=dict(font=dict(
                             # family="sans-serif",
                             # size=15,
                             color="black"
                         ),
                         # text='SNR'
                     ))

    fig.update_traces(marker_line_width=2)
    # fig.add_annotation(dict(font=dict(color="black", size=16,family='times new roman'),
    #                         x=0.95,
    #                         y=0.30,
    #                         showarrow=False,
    #                         # text="Number of IQ samples per signal segment N:",
    #                         xref="paper",
    #                         yref="paper"))
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=110,  # right margin  130
            b=160,  # bottom margin
            t=20,  # top margin
        ),
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        xaxis={"mirror": "all"},
        paper_bgcolor='white',plot_bgcolor='rgba(0,0,0,0)',
        width=500,height=500,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)',orientation='h',
            itemsizing='constant',x=0.35,y=0.25,
            font=dict(size=12, color="black",family='times new roman'),traceorder='normal'
        ),
    )
    plotly.offline.plot(figure_or_data=fig, image_width=610, image_height=500, filename='test_line.html', image='svg')


def plot_constellation():

    trace1 = {
        "mode": "markers+text",
        # "name": "BPSK constellation",
        "type": "scatter",
        "x": [-1, 1],
        "y": [0,0],
        "marker": {"size": [10, 10], "color": '#d62728'},
        # "text": ["0", "1"],
        # "textfont": {"size": 14,"color":"black","family":'Times New Roman'},
        # "textposition": ["top center","top center"],
        "line": {'color':'black','width':12},
    }

    trace2 = {
        "mode": "markers+text",
        # "name": "QPSK constellation",
        "type": "scatter",
        "x": [-0.71, -0.71, 0.71, 0.71],
        "y": [-0.71, 0.71, -0.71, 0.71],
        "marker": {"size": [10, 10, 10, 10], "color": '#d62728'},
        # "text": [" 00", " 01", "10 ", "11 "],
        # "textfont": {"size": 14,"color":"black","family":'Times New Roman'},
        # # "textposition": ["top center"]*4,
        # "textposition": ["middle right", "middle right", "middle left", "middle left"],
        "line": {'color': 'black', 'width': 12},
    }

    i1,q1 = [],[]
    for i in range(-3,4,2):
        for j in range(-3,4,2):
            i1.append(i/math.sqrt(10))
            q1.append(j/math.sqrt(10))

    i2, q2 = [], []
    for i in range(-7, 8, 2):
        for j in range(-7, 8, 2):
            i2.append(i / math.sqrt(42))
            q2.append(j / math.sqrt(42))

    trace3 = {
        "mode": "markers+text",
        # "name": "QPSK constellation",
        "type": "scatter",
        "x": i1,
        "y": q1,
        "marker": {"size": [10]*16, "color": '#d62728'},
        # "text": ["1111", "1110", "1010", "1011",
        #          "1101","1100","1000","1001",
        #          "0101","0100","0000","0001",
        #          "0111","0110","0010","0011"],
        # "textfont": {"size": 14 , "color":"black", "family":'Times New Roman'},
        # "textposition": ["top center"]*16,
        "line": {'color': 'black', 'width': 12},
    }

    trace4 = {
        "mode": "markers+text",
        # "name": "QPSK constellation",
        "type": "scatter",
        "x": i2,
        "y": q2,
        "marker": {"size": [10] * 64, "color": '#d62728'},
        # "text": ["000000"]*64,
        # "textfont": {"size": 8,"color":"black"},
        # "textposition": ["top center"]*64,
        "line": {'color': 'black', 'width': 12},
    }

    shape1 = dict(x0= -1.05,
        x1= 1.05,
        y0= 0,
        y1= 0,
        line=dict(color= "black", dash="dot",width=1),
        type="line",
        xref="x",
        yref="y")

    shape2 = dict(x0= 0,
        x1=0,
        y0=-1.05,
        y1= 1.05,
        line=dict(color= "black", dash="dot",width=1),
        type="line",
        xref="x",
        yref="y")

    # data = [trace1]
    # fig = go.Figure(data)
    fig = plotly.subplots.make_subplots(rows=2, cols=2, horizontal_spacing=0.2,vertical_spacing=0.25)
                                        # subplot_titles=("(a) BPSK, M=1","(b) QPSK, M=2","(c) 16-QAM, M=4",
                                        #                 "(d) 64-QAM, M=6"))
    fig.add_trace(trace1, row=1, col=1)
    fig.add_shape(shape1, row=1, col=1)
    fig.add_shape(shape2, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_shape(shape1, row=1, col=2)
    fig.add_shape(shape2, row=1, col=2)
    fig.add_trace(trace3, row=2, col=1)
    fig.add_shape(shape1, row=2, col=1)
    fig.add_shape(shape2, row=2, col=1)
    fig.add_trace(trace4, row=2, col=2)
    fig.add_shape(shape1, row=2, col=2)
    fig.add_shape(shape2, row=2, col=2)

    row = [1,2]
    col = [1,2]
    for i in row:
        for j in col:
            fig.update_yaxes(showline=True, mirror='all',ticks='inside',
                             linecolor='black', linewidth=1,
                             tickfont=dict(family='Times New Roman',color="black", size=22),
                             tickmode='linear',
                             ticklen=10,
                             tick0=-1.5,
                             dtick=0.5,
                             range=[-1.63, 1.63],
                             # title=dict(font=dict(family='Times New Roman',color="black",  # family="sans-serif",size=15,
                             #                    size=22),
                             #            text='Quadrature',
                             #            standoff=5),
                             row=i, col=j
                             )

            fig.update_xaxes(side='bottom', showline=True, ticks='inside',
                             mirror='all', linecolor='black', linewidth=1,
                             tickfont=dict(family='Times New Roman',color="black", size=22),
                             tickmode='linear',
                             ticklen=10,
                             tick0=-1.5,
                             dtick=0.5,
                             range=[-1.62, 1.62],
                             # title=dict(font=dict(
                             #     family='Times New Roman',
                             #     size=22,
                             #     color="black"
                             # ),
                             #     text='In-phase'),
                             row=i, col=j)



    fig.update_layout(
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        # width=500, height=500,
        showlegend = False,
    )
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20, color='black',family='Times New Roman')
    # image_width=500, image_height=500
    plotly.offline.plot(figure_or_data=fig,image_width=800, image_height=800,filename='const.html', image='svg')


def rcos_freq_response(freq, alpha=0, T=1):
    freq = np.fabs(freq)
    if freq <= (1. - alpha) * 0.5 / T:
        result = T
    elif freq <= (1. + alpha) * 0.5 / T:
        temp = freq - (1 - alpha) * 0.5 / T
        result = 0.5 * T * (1 + np.cos(np.pi * temp * T / alpha))
    else:
        result = 0
    return result



def plot_rc_filter():
    freq = np.linspace(-1.25,1.25,100)
    h_f,h_f1,h_f2=[],[],[]
    for f in freq:
        h_f.append(rcos_freq_response(f,0,1))
        h_f1.append(rcos_freq_response(f, 0.5, 1))
        h_f2.append(rcos_freq_response(f, 1, 1))
    # print(len(f),len(h))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=h_f, name=r'$ \alpha = 0$',
                             line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=freq, y=h_f1, name=r'$ \alpha = 0$',
                             line=dict(color='#2ca02c', width=1)))
    fig.add_trace(go.Scatter(x=freq, y=h_f2, name=r'$ \alpha = 0$',
                             line=dict(color='#ff7f0e', width=1)))
    line_ax = [-1, -0.5, 0, 0.5, 1]
    for i in line_ax:
        fig.add_shape(type="line", x0=i, y0=-0.3, x1=i, y1=1.3,
                      line=dict(
                          color="grey",
                          width=1,
                          dash="dot",
                      ),
                      )
    fig.add_shape(type="line", x0=-1.3, y0=0, x1=1.3, y1=0,
                  line=dict(
                      color="black",
                      width=1,
                  ),
                  )

    fig.update_yaxes(showline=True, mirror='all', ticks='inside',
                    linecolor='black', linewidth=1,
                    tickfont=dict(family='Times New Roman', color="black", size=18),
                    tickmode='linear',
                    ticklen=10,
                    tick0=0,
                    dtick=0.5,
                    range=[-0.3, 1.3],
                    title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                         size=18)))

    fig.update_xaxes(side='bottom', showline=True, ticks='inside',
                     mirror='all', linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     tickmode='linear',
                     ticklen=10,
                     tick0=-1,
                     dtick=0.5,
                     range=[-1.3, 1.3],
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=18,
                         color="black"
                     )))
                         # text=r'$\text{Frequency} f$'))
    fig.update_layout(
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        width=600, height=500,
        showlegend=False,
    )
    plotly.offline.plot(figure_or_data=fig, image_width=400, image_height=350, filename='rcos.html', image='svg')


def plot_rc_impulse():

    alpha = [0,0.5,1]
    count = 0
    marker_colors = ['#1f77b4','#2ca02c','#ff7f0e']
    fig = go.Figure()
    for val in alpha:
        time_idx, h_rc = commpy.rcosfilter(N=1024,alpha=val,Ts=1,Fs=144)
        fig.add_trace(go.Scatter(x=time_idx, y=h_rc,
                                 line=dict(color=marker_colors[count], width=1)))
        count += 1

    line_ax = [-2, -1, 0, 1, 2]
    for i in line_ax:
        fig.add_shape(type="line", x0=i, y0=-0.3, x1=i, y1=1.3,
                      line=dict(
                          color="grey",
                          width=1,
                          dash="dot",
                      ),
                      )
    fig.add_shape(type="line", x0=-2.3, y0=0, x1=2.3, y1=0,
                  line=dict(
                      color="black",
                      width=1,
                  ),
                  )

    fig.update_yaxes(showline=True, mirror='all', ticks='inside',
                     linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     tickmode='linear',
                     ticklen=10,
                     tick0=0,
                     dtick=0.5,
                     range=[-0.3, 1.3],
                     title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                          size=18)))

    fig.update_xaxes(side='bottom', showline=True, ticks='inside',
                     mirror='all', linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     tickmode='linear',
                     ticklen=10,
                     tick0=-2,
                     dtick=1,
                     range=[-2.3, 2.3],
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=18,
                         color="black"
                     )))
    # text=r'$\text{Frequency} f$'))
    fig.update_layout(
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        width=600, height=500,
        showlegend=False,
    )
    plotly.offline.plot(figure_or_data=fig, image_width=400, image_height=350, filename='rcos.html', image='svg')


def norm(x):

    total = []
    max_, min_ = total[0], total[0]
    # print(min_)
    c = 0
    for list in total:

        min_ = [min(value) for value in zip(min_,list)]
        max_ = [max(value) for value in zip(max_,list)]

    norm = [float((x[i]-min_[i]) / (max_[i]-min_[i])) for i in range(len(x))]

    return norm


def sequential_chart():
    path = "/home/rachneet/thesis_results/sequential_intf_bpsk_results1.csv"
    df = pd.read_csv(path, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    # print(df.head())
    groups = df.groupby('True_label')
    names= ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
        "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]

    fig = go.Figure()
    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    # colors = ['#23aaff',  #  blue
    #         'rgb(253,141,60)',  # orange
    #         '#66c56c',  # moss green
    #        'rgb(159,130,206)',  # muted purple
    #         '#f4b247',  # mustard yellow
    #         '#ff6555',  # red apple
    #         'rgb(247,104,161)',  # pink
    #         'rgb(104,171,184)'   # teal
    #           ]
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#bcbd22',  # curry yellow-green
    ]

    line_colors = ['#1f77b4','#ff7f0e','#2ca02c','rgb(130,109,186)',
                   '#ff7f0e','#d62728','rgb(221,52,151)','rgb(79,144,166)']
    x = np.arange(1, 49)
    total = []
    for i in range(len(names)):
        data = groups.get_group(names[i])
        # print(data.head())

        sum_preds = [0]*48

        for row in data.iterrows():
            preds = literal_eval(row[1][1])
            # print(len(preds))
            for j in range(len(preds)):
                sum_preds[j] += preds[j]
        # print(sum_preds)
        # print("------------------------------------")
        #
        # if total != []:
        #     for i in range(len(sum_preds)):
        #         # print("total: ", total)
        #         sum = 0
        #         sum = sum_preds[i] + total[i]
        #         total[i] = sum
        # else:
        #     total.extend(sum_preds)
        # print("total : ", total)

        if names[i] == "SC_BPSK":
            fig.add_trace(go.Bar(x=x, y=norm(sum_preds), orientation='v', name=names[i], marker=dict(
                color=colors[i], opacity=0.8,
                line=dict(color="black", width=2)
            )))
        else:
            fig.add_trace(go.Bar(x=x, y=norm(sum_preds), orientation='v', name=names[i], marker=dict(
                color=colors[i], opacity=0.7,
                line=dict(color=colors[i], width=0)
            )))
    #
    fig.update_yaxes(showline=True, mirror=True, ticks='outside',
                     linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                          size=22),
                                # text="Normalized incorrect<br> predictions"
                                ))

    fig.update_xaxes(side='bottom', showline=True, ticks='outside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     # tick0=5,
                     # dtick=5,
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=22,
                         color="black"
                     ),
                         # text="Signal segments"
                     ))

    fig.update_layout(barmode='relative', # title_text='Incorrect Predictions for Signal Segments',
                      paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False, title_x=0.50, title_y=0.90,
                      margin=dict(b=170, l=0, r=150, t=20),
                      legend=dict(
                          # bordercolor='black',
                          # borderwidth=1,
                          x=0.2, y=1.2,
                          bgcolor='rgba(0,0,0,0)',
                          orientation='h',
                          itemsizing='trace')
                      )

    # 450 700
    plotly.offline.plot(figure_or_data=fig, image_width=1000, image_height=500, filename='seq.html', image='svg')


def draw_activation(x):
    sigmoid, tanh, relu = [],[],[]
    for item in x:
        sigmoid.append(1 / (1 + math.exp(-item)))
        tanh.append(math.tanh(item))

    y = np.arange(-10.0, 10.0, 0.1)
    for val in y:
        relu.append(np.maximum(0,val))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sigmoid, name=""))
    fig.add_trace(go.Scatter(x=x, y=tanh, name=""))
    fig.add_trace(go.Scatter(x=y, y=relu, name=""))

    fig.add_shape(type="line", x0=0, y0=-1.2, x1=0, y1=1.2,
                  line=dict(
                      color="black",
                      width=1
                  ))
    line_ax = [-8, -6, -4, -2, 2, 4, 6, 8]
    for i in line_ax:
        fig.add_shape(type="line", x0=i, y0=-1.2, x1=i, y1=1.2,
                      line=dict(
                          color="grey",
                          width=1,
                          dash="dot",
                      ),
                      )
    fig.add_shape(type="line", x0=-10, y0=0, x1=10, y1=0,
                  line=dict(
                      color="black",
                      width=1
                  ))
    fig.update_yaxes(showline=True, mirror=True, ticks='inside',
                     linecolor='black', linewidth=1,
                     tick0=-1,
                     dtick=0.5,
                     range=[-1.2, 1.2],
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                          size=18)))

    fig.update_xaxes(side='bottom', showline=True, ticks='inside',
                     mirror=True, linecolor='black', linewidth=1,
                     tick0=-10,
                     dtick=2,
                     range=[-10, 10],
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=18,
                         color="black"
                     )))
    fig.update_layout(
        # title_text='<b>Model Scalability with Number of IQ Samples',
        # title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        margin=go.layout.Margin(
            l=0,  # left margin
            r=160,  # right margin  130
            b=160,  # bottom margin
            t=20,  # top margin
        ),
        showlegend=True,
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='v',
            itemsizing='constant',
            # x=0.25,
            x=0.65,
            y=0.1,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            )
    ))
    plotly.offline.plot(figure_or_data=fig, image_width=550, image_height=500, filename='act.html', image='svg')


def deepsig_plots():

    # x = np.arange(-20,21,2)
    x = np.arange(0, 21, 5)
    # x = ['-20 dB', '-10 dB', '0 dB', '10 dB', '20 dB']
    # x = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB']
    # y_xgb = [0.04, 0.04, 0.05, 0.05, 0.06, 0.07, 0.12, 0.19, 0.28, 0.33, 0.43, 0.55, 0.63, 0.67, 0.71,
    #          0.72, 0.73, 0.73, 0.73, 0.73, 0.73]
    # y_cnn = [0.05, 0.05, 0.04, 0.05, 0.07, 0.11, 0.17, 0.23, 0.31, 0.4, 0.5, 0.59, 0.68, 0.76, 0.83,
    #          0.85, 0.87, 0.87, 0.88, 0.88, 0.88]
    # y_res = [0.04, 0.04, 0.05, 0.05, 0.08, 0.13, 0.18, 0.22, 0.31, 0.4, 0.52, 0.61, 0.72, 0.85, 0.92,
    #          0.93, 0.94, 0.95, 0.95, 0.95, 0.95]
    # digital mods
    # y_xgb = [0.14, 0.15, 0.15, 0.17, 0.19, 0.26, 0.38, 0.46, 0.53, 0.62, 0.69, 0.81, 0.91, 0.94, 0.96, 0.97, 0.97, 0.98,
    #          0.98, 0.98, 0.98]
    # y_cnn = [0.15, 0.14, 0.15, 0.17, 0.23, 0.33, 0.43, 0.52, 0.58, 0.71, 0.82, 0.91, 0.96, 0.97, 0.98, 0.98, 0.98, 0.98,
    #          0.98, 0.98, 0.98]
    # y_res = [0.14, 0.14, 0.16, 0.2, 0.26, 0.37, 0.45, 0.51, 0.58, 0.73, 0.84, 0.91, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96,
    #          0.96, 0.97, 0.96]
    # y_tr = [0.25, 0.28, 0.29, 0.34, 0.45, 0.61, 0.68, 0.72, 0.79, 0.86, 0.96, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #         1.0, 1.0, 0.82]
    # y_tl = [0.25, 0.25, 0.25, 0.26, 0.25, 0.28, 0.33, 0.45, 0.53, 0.7, 0.83, 0.88, 0.92, 0.94, 0.97, 0.97, 0.97, 0.97,
    #         0.97, 0.97, 0.97, 0.72]

    # c_lab = [0.608, 0.830, 0.941, 0.986, 0.994]
    # c_sdr = [0.571, 0.785, 0.914, 0.979, 0.991]
    # r_lab = [0.753, 0.980, 1.0, 1.0, 1.0]
    # r_sdr = [0.604, 0.878, 0.977, 0.998, 1.0]

    c_cfo1 = [0.577, 0.801, 0.927, 0.98, 0.992]
    r_cfo1 = [0.753, 0.983, 1, 1, 1]
    c_cfo5 = [0.592, 0.814, 0.937, 0.986, 0.995]
    r_cfo5 = [0.709, 966, 0.999, 1, 1]

    tr = [0.258, 0.251, 0.248, 0.238, 0.257, 0.266, 0.346, 0.47, 0.562, 0.667, 0.775, 0.823, 0.875, 0.94, 0.981, 0.989,
          0.992, 0.991, 0.992, 0.993, 0.993]
    gen = [0.255, 0.254, 0.253, 0.244, 0.257, 0.249, 0.251, 0.247, 0.246, 0.261, 0.254, 0.41, 0.418, 0.246, 0.249,
           0.261, 0.246, 0.25, 0.246, 0.25, 0.248]
    tl = [0.25, 0.253, 0.258, 0.235, 0.255, 0.252, 0.26, 0.31, 0.417, 0.536, 0.696, 0.791, 0.827, 0.894, 0.925, 0.938,
          0.962, 0.948, 0.957, 0.954, 0.955]

    # 'rgba(35, 170, 255, 1)'
    # 'Traditional <br>Learning'  'Transfer <br>Learning'
    colors = ['rgba(255, 101, 85, 1)', 'rgba(35, 170, 255, 1)', 'rgba(244, 178, 71, 1)']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=c_cfo1, mode="lines+markers", marker_symbol='circle', marker_color=colors[0],
                             name='                                          '))
    fig.add_trace(go.Scatter(x=x, y=c_cfo5, mode="lines+markers", marker_symbol='square', marker_color=colors[1],
                             name='                                          '))
    # fig.add_trace(go.Scatter(x=x, y=tl, mode="lines+markers", marker_symbol='diamond', marker_color=colors[2],
    #                          name='                               ',
    #                          # line=dict(
    #                          #     # color="grey",
    #                          #     # width=1,
    #                          #     dash="dashdot",
    #                          # )
    #                          ))

    fig.add_trace(go.Scatter(x=x, y=r_cfo1, mode="lines+markers", marker_symbol='circle', marker_color=colors[0],
                             name='                               ',
                             line=dict(
                                 # color="grey",
                                 # width=1,
                                 dash="dashdot",
                             )
                             ))
    fig.add_trace(go.Scatter(x=x, y=r_cfo5, mode="lines+markers", marker_symbol='square', marker_color=colors[1],
                             name='                               '
                             ))

    line_ax = [0.2, 0.4, 0.6, 0.8, 1]
    # line_ax = [0.6, 0.7, 0.8, 0.9, 1]
    for i in line_ax:
        fig.add_shape(type="line", x0=-20, y0=i, x1=20, y1=i,
                      line=dict(
                          color="grey",
                          width=1,
                          dash="dashdot",
                      ),
                      )

    # line_ax_2 = [-10, 0, 10]
    # line_ax_2 = [5, 10, 15]
    # for i in line_ax_2:
    #     fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=1.05,
    #                   line=dict(
    #                       color="grey",
    #                       width=1,
    #                       dash="dashdot",
    #                   ),
    #                   )

    fig.update_yaxes(showline=True, mirror=True, ticks='outside',
                     linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=16),
                     tick0=0,
                     dtick=0.2,
                     range=[0, 1.05],
                     title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                          size=18),
                                # text="Classification accuracy"
                                ))

    fig.update_xaxes(side='bottom', showline=True, ticks='outside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=16),
                     tick0=-20,
                     dtick=10,
                     range=[-20, 20],
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=18,
                         color="black"
                     ),
                         # text="SNR(dB)"
                     ))

    fig.update_traces(marker_line_width=1)

    fig.update_layout(
        # title_text='Comparison of models for digital modulation dataset',
        title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        # width=600, height=500,
        showlegend=True,
        margin=dict(b=170, l=0, r=100, t=20),   # r=120
        legend=dict(
            # bordercolor='black',
            # borderwidth=1,
            # bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            # y=0.01, x=0.65,
            y=1.2, x=-0.15,
            font=dict(size=12, color="black", family='times new roman'), traceorder='normal'
        )
    )

    plotly.offline.plot(figure_or_data=fig, image_width=600, image_height=450, filename='cfo_comp.html', image='svg')
    # pass


def plot_lpf():
    # filter configuration
    srate = 50000000.0
    bw_signal = 20000000.0
    freq_noise = np.array([5.73e9, 5.77e9])
    bw_noise = 5000000.0
    freq =  5750000000.0
    numtaps = 100
    b = scipy.signal.firwin(numtaps=numtaps,
                            cutoff=bw_signal / 2,
                            pass_zero=True,
                            fs=srate,
                            )

    # show frequency response of the filter

    w, h = scipy.signal.freqz(b, worN=1500)
    w *= srate / (2 * np.pi) / 1e6  # w in MHz
    # print(w)
    y = 20 * np.log10(abs(h))
    y_ph = np.unwrap(np.angle(h)) / np.pi
    #3 dB point
    x1, y1 = w[np.argmin(abs(abs(h) - np.sqrt(0.5)))], 20 * np.log10(np.sqrt(0.5))
    # attenuation f_c
    x2, y2 = bw_signal / 2e6, 20 * np.log10(abs(h[np.argmin(abs(w - bw_signal / 2e6))]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=y_ph, line=dict(
                color='#23aaff',
             )))
    fig.add_trace(go.Scatter(x=[0, 10], y=[5, 5], mode='none', fillcolor='rgba(102, 197, 108, 0.4)', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=[0, 10], y=[-25, -25], mode='none', fillcolor='rgba(102, 197, 108, 0.4)', fill='tozeroy'))
    # fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='markers',marker=dict(
    #         color='#f4b247',
    #         size=6,
    #         line=dict(
    #             color='#ff7f0e',
    #          )
    #     )))# marker_color='#f4b247', marker_line_color='#ff7f0e'))
    fig.update_yaxes(showline=True, mirror=True, ticks='outside',
                     linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     tick0=-20,
                     dtick=10,
                     range=[-25, 5],
                     title=dict(font=dict(family='Times New Roman', color="black",  # family="sans-serif",size=15,
                                          size=18)))

    fig.update_xaxes(side='bottom', showline=True, ticks='outside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(family='Times New Roman', color="black", size=18),
                     tick0=0,
                     dtick=2,
                     range=[0, 25],
                     title=dict(font=dict(
                         family='Times New Roman',
                         size=18,
                         color="black"
                     )))

    fig.update_layout(
        # title_text='Comparison of models for digital modulation dataset',
        # title_x=0.50, title_y=0.90,
        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
        # width=600, height=500,
        showlegend=False,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)', orientation='v',
            itemsizing='constant',
            x=0.01, y=0.99,
            font=dict(size=12, color="black", family='times new roman'), traceorder='normal'
        )
    )

    fig.update_traces(marker_line_width=0.8)
    plotly.offline.plot(figure_or_data=fig, image_width=700, image_height=380, filename='lpf.html', image='svg')
    # pass

    # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 7))
    #
    # for ax, y, label in zip([ax1, ax2],
    #                  [20 * np.log10(abs(h)), np.unwrap(np.angle(h)) / np.pi],
    #                  ["Magnitude [dB]", "Phase [$\\pi^{-1}$]"]):
    #
    #     ax.plot(w, y)
    #     ax.set_xlim([0, min(srate / 2e6, (freq_noise[1] + 2 * bw_noise) / 1e6)])
    #     ax.fill_between([0, bw_signal / 1e6 / 2], -200, 200,
    #                     facecolor=color_bw_signal, label='Signal band')
    #     noise_offset = freq_noise[1] - freq
    #     noise_freq_idx = np.array([(noise_offset - bw_noise / 2) / 1e6,
    #                                (noise_offset + bw_noise / 2) / 1e6])
    #     ax.fill_between(noise_freq_idx, -200, 200,
    #                     facecolor=color_bw_noise, label='Noise band')
    #     ax.set_ylim([min(y), max(y) + 0.1 * abs(min(y))])
    #     ax.set_ylabel(label)
    #     ax.set_title("Frequency response of the low-pass filter")
    #     ax.set_xlabel("Frequency [MHz]")
    #     ax.legend(loc=1)
    #
    # ax1.set_ylim([-80, 5])
    #
    # # -3dB point
    # x, y = w[np.argmin(abs(abs(h) - np.sqrt(0.5)))], 20 * np.log10(np.sqrt(0.5))
    # ax1.plot(x, y, 'ob', markersize=4, color='red')
    # ax1.annotate('%5.2fdB @\n%5.2f MHz' % (y, x),
    #     xy=(x, y),
    #     xytext=(10, 0),
    #     textcoords='offset points',
    #     ha='left',
    #     va='center')
    #
    # # attenuation @ f_c
    # x, y = bw_signal / 2e6, 20 * np.log10(abs(h[np.argmin(abs(w - bw_signal / 2e6))]))
    # ax1.plot(x, y, 'og', markersize=4, color='red')
    # ax1.annotate('%5.2fdB @\n%5.2f MHz' % (y, x),
    #     xy=(x, y),
    #     xytext=(-10, -5),
    #     textcoords='offset points',
    #     ha='right',
    #     va='center')
    #
    # # average noise attenuation
    # noise_idx = np.array(np.arange(noise_freq_idx[0] * (w.size / (srate / 2e6)),
    #                                noise_freq_idx[1] * (w.size / (srate / 2e6)), 1), dtype=int)
    # avg_mag_noise = 10 * np.log10(np.mean(abs(h[noise_idx])**2))
    # ax1.plot(noise_freq_idx, [avg_mag_noise] * 2, 'r')
    #
    # x, y = np.mean(noise_freq_idx), avg_mag_noise
    # ax1.annotate('average noise\nattenuation: %5.2fdB' % avg_mag_noise,
    #     xy=(x, y),
    #     xytext=(25, 30),
    #     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8, headlength=8),
    #     textcoords='offset points',
    #     ha='center',
    #     va='bottom')
    #
    # plt.tight_layout()
    # # plt.show()
    # # plt.show()
    # plt.savefig("test.svg")


if __name__=="__main__":
    plot_line_chart()
    # plot_lpf()
    # plot_barchart()
    # train_fraction_effect()
    # x = np.arange(-10,10,0.2)
    # draw_activation(x)
    # draw_comparison_chart()
    # scalability_chart()
    # n = 4
    # plot_barchart()
    # plot_heatmap()
    # plt_tl_chart()
    # for i in range(16):
    #     b = bin(i)[2:].zfill(n)
    #     print(b)
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
    # sequential_chart()
    # norm(0)
    # pass