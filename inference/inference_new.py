# inference module for cnn
from torch.autograd import Variable
from data_processing.dataloader import *
from models.pytorch.train import get_evaluation
import csv
import pandas as pd
from copy import deepcopy
# torch.cuda.set_device(0)


# plotting libraries
import plotly
import plotly.figure_factory as ff
import plotly.io as pio
plotly.io.orca.config.save()
pio.renderers.default = 'svg'

def evaluate(y_true, y_pred, list_metrics):

    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    return output


def compute_results(csv_path):
    """
    Input: path to the output csv file you get from the inference function
    Returns:
        Dict : Counts of signals in each SNR category
        Dict : Accuracy and Confusion matrices for all SNRs
    """

    df = pd.read_csv(csv_path, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    # print(df.head())
    groups = df.groupby('SNR')
    snrs = [5, 10, 15, 20, 25]
    total_count,result = {},{}

    for snr in snrs:
        data = groups.get_group(snr).reset_index(drop=True)
        output = evaluate(data['True_label'].values, data['Predicted_label'].values,
                          ['accuracy', 'confusion_matrix'])
        unique, counts = np.unique(data['True_label'].values, return_counts=True)
        result[snr] = output
        total_count[snr] = counts

    return total_count,result


def inference(save_path,x_test_gen,y_test_gen,y_test_raw,snr_gen,model_name):

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


    _labels =[]
    for _, l in enumerate(y_test_raw):
        # print(x)
        _labels.append(label_idx(l))

    unique, counts = np.unique(_labels, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    output_file = open(save_path + "test_logs.txt", "w")
    model = torch.load(save_path + model_name, map_location='cuda:0')
    # ica = FastICA(n_components=256,tol=1e-5,max_iter=1000)
    model.eval()

    with torch.no_grad():

        test_true = []
        test_prob = []
        snr_vals = []

        for batch in zip(x_test_gen,y_test_gen,snr_gen):
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

    fieldnames = ['True_label', 'Predicted_label', 'SNR']
    with open(save_path + "output.csv", 'w',encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(np.argmax(test_true, -1), np.argmax(test_prob, -1), snr_vals):
            writer.writerow(
                {'True_label': i, 'Predicted_label': j, 'SNR': k})

    test_metrics = get_evaluation(test_true, test_prob,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    output_file.write(
        "Test loss: {} Test accuracy: {} \nNum Samples per class: \n{} \nTest confusion matrix: \n{}\n\n".format(
            test_metrics["loss"],
            test_metrics["accuracy"],
            np.asarray((unique, counts)).T,
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

    fig.update_layout(title_text='<b>CNN Performance with the presence of Interfering OFDM Signals <br> at SIR ' + str(snr) + 'dB </b>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.18,
                            showarrow=False,
                            text="Predicted class",
                            xref="paper",
                            yref="paper"))

    # add colorbar
    fig['data'][0]['showscale'] = True

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.20,
                            y=0.5,
                            showarrow=False,
                            textangle=-90,
                            text="True class",
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200),
                      title_x=0.55,
                      title_y=0.97,
                      paper_bgcolor='white',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

    fig['data'][0].colorbar = dict(title='Number of <br>Samples',
                                   outlinecolor="black",
                                   outlinewidth=1,
                                   ticks='outside',
                                   len=1.05
                                   )

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     title=dict(
                         font=dict(
                             family="sans-serif",
                             size=18,
                             # color="#7f7f7f"
                         ))
                     )
    fig.update_xaxes(automargin=True, tickangle=-90, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     linecolor='black',
                     linewidth=0.5,
                     title=dict(
                         font=dict(
                             family="sans-serif",
                             size=18,
                             # color="#7f7f7f"
                         ))
                     )

    # fig.show()
    plotly.offline.plot(fig, filename=fig_name + ".html", image='svg')


if __name__ == "__main__":
    data_path = "/home/rachneet/rf_dataset_inets/dataset_deepsig_dig_mod.hdf5"
    save_path = "/home/rachneet/thesis_results/deepsig_dig_mod/"
    model = save_path + "model"
    x_train_gen, y_train_gen, x_val_gen, y_val_gen,x_test_gen, y_test_gen,y_test_raw,snr_test_gen =\
        load_batch(data_path,512,"both")
    inference(save_path,x_test_gen,y_test_gen,y_test_raw,snr_test_gen,model)
