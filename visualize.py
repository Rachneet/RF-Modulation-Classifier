import numpy as np
from Receiver import *
from dataloader import *
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

# plotting libraries
import plotly
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


def plot_barchart():
    sirs = ['5 dB','10 dB','15 dB','20 dB','25 dB']
    # acc = [0.82,0.86,0.87,0.88,0.89]
    # acc2 = [0.83,0.86,0.86,0.87,0.87]
    # acc3 = [0.71,0.80,0.86,0.87,0.87]
    acc = [0.72,0.76,0.79,0.80,0.80]
    acc2 = [0.75,0.80,0.80,0.81,0.81]
    acc3 = [0.69,0.78,0.80,0.82,0.82]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        name='SC_BPSK',
        text=acc,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#ff6555',
        marker_line_color='#d62728'
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        name='SC_16QAM',
        text=acc2,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#66c56c',
        marker_line_color='#2ca02c'
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc3,
        name='OFDM_64QAM',
        text=acc3,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#f4b247',
        marker_line_color='#ff7f0e'
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
                     range=[0,1],
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                     text='Accuracy')
                     )
    fig.update_xaxes(automargin=True, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                     text='SIR'
                     )
                     )
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=2, opacity=0.8)

    fig.update_layout(
        title_text='<b>CNN Performance with Interfering Signals <br>at SNR 10 dB',
        title_x=0.50,
        title_y=0.95,
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
                                   tick0=0,
                                   dtick=0.1,
                                   len=1.05
                                   )
    fig.update_layout(title_text='<b>CNN Performance <br>with Transfer Learning Approach',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )
    # add custom colorbar title
    fig.add_annotation(dict(font=dict(color="black"), #size=14),
                            x=1.13,
                            y=1.07,
                            showarrow=False,
                            text="Accuracy",
                            xref="paper",
                            yref="paper"))

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     color='black',
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
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=18,
                             color="black",
                         )),
                     )
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=1.15,
                            showarrow=False,
                            text="SNR (Test)",
                            xref="paper",
                            yref="paper"))
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.15,
                            y=0.5,
                            showarrow=False,
                            textangle=-90,
                            text="SNR (Train)",
                            xref="paper",
                            yref="paper"))
    fig.update_layout(
        margin=dict(b=100, l=150, r=150),
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
    sirs = ['5 dB', '10 dB', '15 dB', '20 dB', '25 dB']

    acc = [0.69,0.78,0.80,0.82,0.82]
    acc2 = [0.78,0.86,0.87,0.87,0.87]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        name='CNN (9-Layer)',
        text=acc,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#ff6555',
        marker_line_color='#d62728',
        # width=0.25,
        # offset=-0.32
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        name='ResNet-101',
        text=acc2,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#66c56c',
        marker_line_color='#2ca02c',
        # width=0.25,
        # offset=0.02
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
                     range=[0, 1],
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         text='Accuracy')
                     )
    fig.update_xaxes(automargin=True, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         text='SIR'
                     )
                     )
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=2, opacity=0.8)

    fig.update_layout(
        title_text='<b>Model Comparison with Interfering OFDM Signals <br>at SNR 10 dB',
        title_x=0.50,
        title_y=0.90,
        yaxis={"mirror": "all"},
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.1,
        width=500,
        height=500,
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

    plotly.offline.plot(figure_or_data=fig,image_width=600, image_height=500, filename='test_bar.html', image='svg')


def plt_tl_chart():
    sirs = ['5 dB', '10 dB', '15 dB', '20 dB', '25 dB']
    # acc = [0.82,0.86,0.87,0.88,0.89]
    # acc2 = [0.83,0.86,0.86,0.87,0.87]
    # acc3 = [0.71,0.80,0.86,0.87,0.87]
    acc = [0.69,0.78,0.80,0.82,0.82]
    acc2 = [0.66,0.76,0.79,0.80,0.80]

    fig = go.Figure()
    # fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]], print_grid=True)
    fig.add_trace(go.Bar(
        x=sirs,
        y=acc,
        name='Traditional <br>Learning',
        text=acc,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#66c56c',
        marker_line_color='#2ca02c',
        # width=0.25,
        # offset=-0.32
    ))

    fig.add_trace(go.Bar(
        x=sirs,
        y=acc2,
        name='Transfer <br>Learning (TL)',
        text=acc2,
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#f4b247',
        marker_line_color='#ff7f0e'
        # width=0.25,
        # offset=0.02
    ))

    fig.add_trace(go.Scatter(
        x=sirs,
        y=[0.23]*5,
        name='Training time <br> with TL',
        mode='lines+markers',
        # text=acc2,
        # textfont=dict(color='black', size=10),
        # textposition='auto',
        marker_color='rgba(5, 112, 176, .8)',
        # marker_color='#f4b247',
        # marker_line_color='#ff7f0e',
        yaxis='y2'
        # width=0.25,
        # offset=0.02
    ))

    fig.add_trace(go.Scatter(
        x=sirs,
        y=[3.71] * 5,
        name='Training time <br>w/o TL',
        mode='lines+markers',
        # text=acc2,
        # textfont=dict(color='black', size=10),
        # textposition='auto',
        marker_color='rgba(152, 0, 0, .8)',
        # marker_color='#66c56c',
        # marker_line_color='#2ca02c',
        yaxis='y2'
        # width=0.25,
        # offset=0.02
    ))

    fig.update_yaxes(automargin=True,showline=True,ticks='inside',mirror=True,
                     linecolor='black',linewidth=1,tickmode='linear',tick0=0,
                     dtick=0.1,range=[0, 1],title=dict(font=dict(# family="sans-serif",
                             # size=15,
                             color="black"
                         ),
                         text='Accuracy')
                     )

    fig.update_xaxes(automargin=True, side='bottom',showline=True,ticks='outside',
                     mirror=True,linecolor='black',linewidth=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             # size=15,
                             color="black"
                         ),
                         text='SIR'
                     ))
    # Customize aspect
    # marker_line_color = '#d62728'

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=2, opacity=0.8)

    fig.update_layout(
        title_text='<b>Comparison of Learning Approaches <br>with Interfering OFDM Signals at SNR 10 dB',
        title_x=0.50,
        title_y=0.90,
        # yaxis={"mirror": "all"},
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.1,
        width=500,
        height=500,
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
        ),
        yaxis2=dict(
        title='Training Time (Hours)',
        title_font=dict(color='black'),
        side='right',
        tickmode='linear',
        tick0=0,
        ticks='inside',
        dtick=1,
        range=[0,10],
        overlaying= 'y'
    ),
    )

    plotly.offline.plot(figure_or_data=fig, image_width=610, image_height=500, filename='test_bar.html', image='svg')


def scalability_chart():
    snrs = ['0 dB', '5 dB', '10 dB', '15 dB', '20 dB']
    acc_256 = [0.463,0.635,0.778,0.882,0.913]
    acc_512 = [0.519,0.721,0.870,0.960,0.978]
    acc_1024 = [0.571,0.785,0.915,0.980,0.991]
    acc_2048 = [0.643,0.864,0.972,0.997,0.999]
    y_dict = {'256': acc_256, '512': acc_512, '1024': acc_1024, '2048': acc_2048}

    fig = go.Figure()
    # fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]], print_grid=True)
    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    names = ['256','512','1024','2048']
    colors = ['rgba(255, 101, 85, 1)','rgba(35, 170, 255, 1)','rgba(244, 178, 71, 1)','rgba(102, 197, 108, 1)']

    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=snrs,y=y_dict[names[i]],name=names[i],
            mode='lines+markers',marker_color=colors[i],
            # marker_color='#f4b247', marker_line_color='#ff7f0e',
            # width=0.25
        ))

    line_ax = [0.5,0.6,0.7,0.8,0.9,1]
    for i in line_ax:
        fig.add_shape(type="line",x0=-0.3,y0=i,x1=5,y1=i,
            line=dict(
                color="grey",
                width=1,
                dash="dashdot",
            ),
        )

    fig.update_yaxes(automargin=True, showline=True, ticks="", mirror=True,
                     linecolor='black', linewidth=1,
                     tickfont=dict(color="black"),
                     title=dict(font=dict( color="black",# family="sans-serif",size=15,
                    ),
                    text='Accuracy')
                    )

    fig.update_xaxes(automargin=True, side='bottom', showline=True, ticks='inside',
                     mirror=True, linecolor='black', linewidth=1,
                     tickfont=dict(color="black"),
                     title=dict(font=dict(
                             # family="sans-serif",
                             # size=15,
                             color="black"
                         ),
                         text='SNR'
                     ))

    fig.update_traces(marker_line_width=2)
    fig.add_annotation(dict(font=dict(color="black", size=10),
                            x=0.9,
                            y=0.30,
                            showarrow=False,
                            text="Number of IQ samples per signal segment N:",
                            xref="paper",
                            yref="paper"))
    fig.update_layout(
        title_text='<b>Model Scalability with Number of IQ Samples',
        title_x=0.50, title_y=0.90,xaxis={"mirror": "all"},
        paper_bgcolor='white',plot_bgcolor='rgba(0,0,0,0)',
        width=500,height=500,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)',orientation='h',
            itemsizing='constant',x=0.35,y=0.25,
            font=dict(size=10, color="black"),traceorder='normal'
        ),
    )
    plotly.offline.plot(figure_or_data=fig, image_width=610, image_height=500, filename='test_line.html', image='svg')


if __name__=="__main__":
    scalability_chart()
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
    # pass