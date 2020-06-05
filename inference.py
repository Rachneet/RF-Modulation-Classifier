# inference module for cnn
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from cnn_model import *
from dataloader import *
from train import *
import csv
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
torch.cuda.set_device(0)


# plotting libraries
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
plotly.io.orca.config.save()
pio.renderers.default = 'svg'

from lightning_resnet import *

def evaluate(y_true, y_pred, list_metrics):

    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    return output


def compute_results(csv_path, snrs=[5, 10, 15, 20, 25]):
    """
    Input: path to the output csv file you get from the inference function
    Returns:
        Dict : Counts of signals in each SNR category
        Dict : Accuracy and Confusion matrices for all SNRs
    """

    df = pd.read_csv(csv_path, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    # print(df.head())
    groups = df.groupby('SNR')
    snrs = snrs
    total_count,result = {},{}

    for snr in snrs:
        data = groups.get_group(snr).reset_index(drop=True)
        output = evaluate(data['True_label'].values, data['Predicted_label'].values,
                          ['accuracy', 'confusion_matrix'])
        unique, counts = np.unique(data['True_label'].values, return_counts=True)
        result[snr] = output
        total_count[snr] = counts

    return total_count,result

# datapath,x_test_gen,y_test_gen,y_test_raw,snr_gen,model_name


def inference(datapath, test_set, model_name, save_path):

    # x_test,y_test,labels_raw = load_batch(datapath+"vsg_no_intf_sc_normed.h5",batch_size=batch_size,mode='test')
    # iq, labels, snrs = reader.read_hdf5(path)
    # test_bound = int(0.80 * labels.shape[0])
    # training_params = {"batch_size": batch_size,
    #                    "shuffle": False,
    #                    "num_workers": 4}
    # x_test_gen = DataLoader(iq[test_bound:].values, **training_params)
    # y_test_gen = DataLoader(labels[test_bound:].values, **training_params)
    # snr_gen = DataLoader(snrs[test_bound:].values, **training_params)
    # y_test_raw = labels[test_bound:].values
    # print("Data Loaded...")


    # _labels =[]
    # for _, l in enumerate(y_test_raw):
    #     # print(x)
    #     _labels.append(label_idx(l))
    #
    # unique, counts = np.unique(_labels, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = open(save_path+"logs.txt", "w")
    # torch.nn.Module.dump_patches=True
    model = torch.load(datapath+model_name, map_location='cuda:0')
    model.eval()

    with torch.no_grad():

        test_true = []
        test_prob = []
        snr_vals = []

        for batch in tqdm(test_set):
            _, n_true_label,snr = batch
            true_label_copy = deepcopy(n_true_label)
            test_true.extend(true_label_copy.cpu().data.numpy())
            del n_true_label
            snr_copy = deepcopy(snr)
            snr_vals.extend(snr_copy.cpu().data.numpy())
            del snr

            batch = [Variable(record).cuda() for record in batch]

            t_data, _, _ = batch
            # x = t_data.view(-1, t_data.shape[1] * t_data.shape[2])
            # input = ica.fit_transform(x.cpu())
            # input = torch.Tensor(input).cuda()
            # input = input.view(-1, 128, 2)
            t_predicted_label = model(t_data)

            test_prob.append(t_predicted_label)

        test_prob = torch.cat(test_prob, 0)
        test_prob = test_prob.cpu().data.numpy()
        # test_true = np.array(test_true)
        # test_pred = np.argmax(test_prob, -1)

    fieldnames = ['True_label', 'Predicted_label', 'SIR']
    with open(save_path + "output.csv", 'w',encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(np.argmax(test_true, -1), np.argmax(test_prob, -1), snr_vals):
            writer.writerow(
                {'True_label': i, 'Predicted_label': j, 'SIR': k})

    test_metrics = get_evaluation(test_true, test_prob,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    output_file.write(
        "Test loss: {} Test accuracy: {}  \nTest confusion matrix: \n{}\n\n".format(
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["confusion_matrix"]))
    output_file.close()

    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


def plot_confusion_matrix(cmap,num_samples,fig_name,snr):
    # cmap = [[30734, 21, 4, 2, 0, 0, 0, 0],
    #         [258, 25387, 3451, 1562, 0, 0, 1, 0],
    #         [41, 5030, 6511, 19180, 0, 0, 3, 1],
    #         [26, 3362, 5378, 22043, 1, 0, 2, 1],
    #         [2, 0, 0, 2, 8405, 12777, 7898, 1656],
    #         [0, 0, 0, 0, 8277, 12763, 7924, 1838],
    #         [1, 0, 0, 0, 754, 4683, 11411, 13623],
    #         [0, 0, 0, 0, 130, 1528, 7230, 21859]]
    #
    # num_samples = [30761, 30659, 30766, 30813, 30740, 30802, 30472, 30747]

    temp2 = []
    for i in range(len(cmap)):
        temp = []
        for j in range(len(num_samples)):
            temp.append('%.2f' % (cmap[i][j] / num_samples[i]))
        temp2.append(temp)


    # print(temp2)

    x = ["SC <br>BPSK", "SC <br>QPSK", "SC 16-<br>QAM", "SC 64-<br>QAM",
         "OFDM <br>BPSK", "OFDM <br>QPSK", "OFDM 16-<br>QAM", "OFDM 64-<br>QAM"]

    y = list(reversed(x))

    # fig = px.imshow(cmap)

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in np.array(temp2[::-1])]

    # print(z_text)
    z2 = []
    for val_list in z_text:
        z1 = []
        for element in val_list:
            z1.append(element + "(1.00)")
        z2.append(z1)

    # set up figure
    colorscale = [[0, 'white'], [1, 'rgb(235,74,64)']]
    # font_colors = ['white', 'black']
    fig = ff.create_annotated_heatmap(list(reversed(cmap)), x=x, y=y, annotation_text=z_text,
                                      colorscale=colorscale)

    # add title

    # fig.update_layout(title_text='<b>Resnet Performance with the presence of Interfering BPSK Signals <br> at SIR ' + str(snr) + 'dB </b>',
    #                   # xaxis = dict(title='x'),
    #                   # yaxis = dict(title='x')
    #                   )

    # add custom xaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=0.5,
    #                         y=-0.18,
    #                         showarrow=False,
    #                         #text="Predicted class",
    #                         xref="paper",
    #                         yref="paper"))

    # add colorbar
    fig['data'][0]['showscale'] = True

    # add custom yaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=-0.20,
    #                         y=0.5,
    #                         showarrow=False,
    #                         textangle=-90,
    #                        # text="True class",
    #                         xref="paper",
    #                         yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200),
                      title_x=0.55,
                      title_y=0.97,
                      paper_bgcolor='white',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

    fig['data'][0].colorbar = dict(#title='Number of Samples',
                                   outlinecolor="black",
                                   outlinewidth=1,
                                   ticks='outside',
                                   len=1.075,  #1.05 ow
                                    # y=0.47,
                                   tickfont=dict(color='black')
                                   )

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     tickfont=dict(family="times new roman",color='black'),
                     title=dict(
                         font=dict(
                             family="times new roman",
                             size=20,
                             color="black"
                         ))
                     )
    fig.update_xaxes(automargin=True, tickangle=-90, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     tickfont=dict(family="times new roman",color='black'),
                     title=dict(
                         font=dict(
                             family="times new roman",
                             size=20,
                             color="black"
                         ))
                     )

    # fig.show()
    plotly.offline.plot(fig, filename=fig_name + ".html", image='svg',image_height=400, image_width=600,
                        include_plotlyjs=True)



def test_sequential(path):
    df = pd.read_csv(path+"output.csv", delimiter=",", quoting=csv.QUOTE_MINIMAL)
    # print(df.tail())
    # print(df.shape)
    mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                   "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
    # combine signals to form one large segment
    count, sig_num = 0,0
    signals = dict()
    t_label, p_label, sir = [], [], []
    for row in df.iterrows():
        count += 1
        # print(row[1][2])
        t_label.append(row[1][0])
        p_label.append(row[1][1])
        sir.append(row[1][2])

        if count % 48 == 0:
            sig_num += 1
            signals[sig_num] = [t_label, p_label, sir]
            # print(len(t_label))
            t_label, p_label, sir = [], [], []


        # if count==20:
        #     print(signals)
        #     break
    print(len(signals))
    print("====================Signals created====================")

    # analyse correct predictions : get their indices
    predictions = {}
    for k,v in signals.items():
        info = {}
        incorrect_idx = []
        # print(v[1])

        for i in range(len(v[0])):
            if v[1][i] != v[0][i]:
                incorrect_idx.append(i+1)
        # print(mod_schemes[v[1][0]])
        # print(incorrect_idx)
        if len(incorrect_idx) == 0:
            incorrect_idx = 0
        info['incorrect_idx'] = incorrect_idx
        info['True_label'] = mod_schemes[v[1][0]]
        info['sir'] = v[2][0]
        predictions[k] = info



    output = pd.DataFrame.from_dict(predictions, orient='index')

    # remove entries where all predictions are correct
    output = output[output.incorrect_idx != 0]
    output = output.sample(frac=1).reset_index(drop=True)
    # print(output.head(20))
    # print(output.head(50))
    sirs = [5,10,15,20,25]
    # labels = [0,1,2,3,4,5,6,7]
    groups = output.groupby(['True_label','sir'])
    out_file = open("/home/rachneet/thesis_results/sequential_intf_bpsk_results.csv","w")
    fieldnames = ["True_label", "Incorrect_preds", "SIR"]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()

    for sir in sirs:
        for mod in mod_schemes:
            idx = []
            try:
                data = groups.get_group((mod,sir)).reset_index(drop=True)
                for row in data.iterrows():
                    # print(row[1][0])
                    idx.extend(row[1][0])
                unique, counts = np.unique(idx, return_counts=True)
                # print(unique, counts)
                # print("====================Processing final counts========================")
                counts = counts.tolist()
                for i in range(48):
                    if (i+1) not in unique and i < len(counts):
                        counts.insert(i,0)
                    elif (i+1) not in unique and i >= len(counts):
                        counts.append(0)

                writer.writerow({"True_label": mod,"Incorrect_preds": counts,"SIR": sir})

                print(mod, sir)
                print(unique, counts)
                # break

            except Exception as e:
                print(repr(e))


        #     break
        # break
    out_file.close()




if __name__ == "__main__":
    pass
    # path = "/home/rachneet/thesis_results/res_sequential_test_intf_bpsk/"
    # test_sequential(path)
    # path = "/home/rachneet/thesis_results/sequential_intf_bpsk_results.csv"
    # df = pd.read_csv(path, delimiter=",", quoting=csv.QUOTE_MINIMAL)
 #    cmap = [[1268  ,   0  ,   0 ,    0 ,    0 ,    0  ,   0  ,   0],
 # [    0 ,1186 ,    0 ,    0,     0,     0 ,    0 ,    0],
 # [    0  ,   0, 1233 ,  0 ,    0   ,  0   ,  0 ,    0],
 # [    0  ,   0 ,   0 ,1266,     0 ,    0  ,   0,     0],
 # [    0 ,    0 ,    0 ,    0 ,882 ,   128 ,    87  ,   143],
 # [    0   ,  0  ,   0  ,   0 ,  81 ,685,   297 ,   168],
 # [    0  ,   0 ,    0   ,  0  ,  65 ,  270, 771,  205],
 # [    0  ,   0  ,   0   ,  0  ,  112 ,  259 ,  126 ,768]]
 #
 #    counts=[]
 #    for i in cmap:
 #        counts.append(sum(i))
 #    # counts = [30761,30659,30766,30813,30740,30802,30472,30747]
 #    # print(counts)
 #
 #    plot_confusion_matrix(cmap, counts, "cmap", "")

    # ---------------------------------------MAIN-------------------------------------------------------
    # training_params = {'batch_size':512, 'num_workers':10}
    # datapath = "/media/rachneet/arsenal/rf_dataset_inets/dataset_intf_free_vsg_cfo5_all.h5"
    # _, _, test_set = load_data(datapath, 0, 0.2, **training_params)
    # # print(test_set)
    # # for batch in test_set:
    # #     _,labels,_ = batch
    # #     print(labels)
    # #     break
    # # file = h5.File(datapath,'r')
    # # labels = file['labels']
    # # print(labels[:50])
    # inf_path = "/home/rachneet/thesis_results/"
    # save_path = inf_path+"cnn_train_cfo1_test_cfo5/"
    # inference(inf_path, test_set, "trained_cnn_vsg_cfo1_all",save_path)
    # pass

    # -------------PLot overall conf maps------------------------------------------------------------
    datapath = "/home/rachneet/thesis_results/"
    file = datapath+'cnn_train_cfo5_test_cfo1/output.csv'
    df = pd.read_csv(file)
    # print(df.tail())
    output = {}
    y_true = df['True_label'].values
    y_pred = df['Predicted_label'].values
    print(metrics.accuracy_score(y_true, y_pred))
    # cmap =  metrics.confusion_matrix(y_true, y_pred)
    # # print(cmap)
    # unique, counts = np.unique(df['True_label'].values, return_counts=True)
    # # print(counts)
    # # k=25
    # plot_confusion_matrix(cmap, counts, "cmap_intf_bpsk_sir_" + "all", "all")

    # -------------PLot individual conf maps------------------------------------------------------------
    # datapath = "/home/rachneet/thesis_results/"
    # file = datapath + 'intf_bpsk_snr10_all/output.csv'
    # df = pd.read_csv(file)
    # # print(df.tail())
    # count,output = compute_results(file,[20])
    # for k, v in output.items():
    #     plot_confusion_matrix(v['confusion_matrix'],count[k],"cmap"+str(k),k)
    #     # print(v['confusion_matrix'])
    #     # print(v['accuracy'])

    # -------------------Plot collective conf maps-----------------------------------------------------

    # count,output = compute_results(datapath+"cnn_train_cfo5_test_cfo1/output.csv", [0,5,10,15,20])
    # # print(count,output)
    # # print(count)
    # for k,v in output.items():
    #     # plot_confusion_matrix(v['confusion_matrix'],count[k],"cmap_intf_bpsk_snr_"+str(k),k)
    #     # print(v['confusion_matrix'])
    #
    #     print(v['accuracy'])

    # --------------------------------CFO correction--------------------------------------------------

    # file = datapath+"output_cnn_vsg_cfo5_all_test.csv"
    # df = pd.read_csv(file)
    # df.drop(columns=['SNR'],inplace=True)
    # df2 = pd.read_csv(datapath+"snr_cfo5.csv",names=['SNR'])
    # df['SNR'] = df2.SNR.values
    # df.to_csv(datapath+'output_cnn_vsg_cfo5_all.csv',index=False)
    # print(df.head())
    # print(df2.head())

# ------------------------------------------------------------------------------------------------------------



