# inference module for cnn
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from cnn_model import *
from dataloader import *
from train import *
import csv
from copy import deepcopy
# torch.cuda.set_device(0)


# plotting libraries
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
plotly.io.orca.config.save()
pio.renderers.default = 'svg'



def inference(datapath,x_test_gen,y_test_gen,y_test_raw,snr_gen,model_name):

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

    output_file = open("test_cnn_no_intf_usrp_all_logs.txt", "w")
    model = torch.load(model_name)
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
            t_predicted_label = model(t_data)

            test_prob.append(t_predicted_label)

        test_prob = torch.cat(test_prob, 0)
        test_prob = test_prob.cpu().data.numpy()
        # test_true = np.array(test_true)
        # test_pred = np.argmax(test_prob, -1)

    fieldnames = ['True label', 'Predicted label', 'SNR']
    with open(datapath + "output_usrp_all_test.csv", 'w',encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(np.argmax(test_true, -1), np.argmax(test_prob, -1), snr_vals):
            writer.writerow(
                {'True label': i, 'Predicted label': j, 'SNR': k})

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


if __name__ == "__main__":
    # inference(datapath="/media/backup/Arsenal/rf_dataset_inets/",batch_size=512)
    # pass
    datapath = "/media/backup/Arsenal/rf_dataset_inets/vsg_no_intf_sc_normed.h5"
    iq, labels, snrs = reader.read_hdf5(datapath)
    test_bound = int(0.80 * labels.shape[0])
    training_params = {"batch_size": 256,
                       "shuffle": False,
                       "num_workers": 4}
    x_test_gen = DataLoader(iq[test_bound:], **training_params)
    y_test_gen = DataLoader(labels[test_bound:], **training_params)
    snr_gen = DataLoader(snrs[test_bound:], **training_params)
    y_test_raw = labels[test_bound:]
    print("Data Loaded...")

    # _labels = []
    # for _, l in enumerate(y_test_raw):
    #     # print(x)
    #     _labels.append(label_idx(l))

    # unique, counts = np.unique(_labels, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    # output_file = open("test_cnn_no_intf_vsg_sc_all_logs.txt", "w")
    # model = torch.load("trained_cnn_no_intf_vsg_sc_all")
    # model.eval()

    # with torch.no_grad():
    #
    #     test_true = []
    #     test_prob = []
    #     snr_vals = []

    for batch in zip(x_test_gen, y_test_gen, snr_gen):
        _, n_true_label, snr = batch



