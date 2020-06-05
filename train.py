import cnn_model
import cnn_unk
import resnet
# from image_dataloader import image_dataloader
import resnet_simplified
import dataloader as dl
import read_h5 as reader
import inference as inf
import visualize
import scipy.spatial.distance as spd
from read_filtered import sort_matrix_entries

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from sklearn.decomposition import FastICA
from torch.autograd import Variable
from functools import partial
from collections import defaultdict
from sklearn import preprocessing
# torch.cuda.set_device(1)

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)


def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a,b)
    return di


def train(data_path,model,num_epochs):

    # x_test,y_test,raw_lables,snr_gen
    x_train,y_train,x_val,y_val = \
        dl.load_batch("/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5"
                      ,512,mode='train')  #GOLD_XYZ_OSC.0001_1024.hdf5
    # y_train = torch.from_numpy(y_train).view(-1, 1)
    # y_val = torch.from_numpy(y_val).view(-1, 1)
    # path = "/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5"
    # iq, labels, snrs = reader.read_hdf5(path)
    # x_train = DataLoader(iq,batch_size=2)
    # y_train = DataLoader(labels,batch_size=2)

    print("Data loaded and batched...")
    # model = cnn_model.CNN(n_classes=8)
    # model = resnet.resnet50(2,24)
    # model = resnet_simplified.ResNet50(n_classes=8)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # regularization
    # l2_reg = None
    # for w in model.parameters():
    #     if l2_reg is None:
    #         l2_reg = w.norm(2)
    #     else:
    #         l2_reg = l2_reg+w.norm(2)

    num_iter_per_epoch = len(x_train)
    best_accuracy = 0
    # reg_lambda = 0.1

    output_file = open(data_path+"tl_vsg_15_20_logs.txt", "w")
    # ica = FastICA(n_components=256,tol=1e-5,max_iter=1000)
    # print(zip(x_train_gen, y_train_gen))
    # activations = visualize.SaveFeatures(list(model.children())[5])
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        average_loss = 0

        for iter, batch in enumerate(zip(x_train,y_train)):

            _, n_mod = batch

            batch = [Variable(record).cuda() for record in batch]
            optimizer.zero_grad()
            t_iq, t_mod = batch

            # # perform blind source separation
            # x = t_iq.view(-1, t_iq.shape[1] * t_iq.shape[2])
            # input = ica.fit_transform(x.cpu())
            # input = torch.Tensor(input).cuda()
            # input = input.view(-1,128,2)
            pred = model(t_iq)
            n_prob_label = pred.cpu().data.numpy()
            # print("Model Prediction: {}".format(n_prob_label))
            loss = criterion(pred, torch.max(t_mod,1)[1])   # + l2_reg*reg_lambda
            running_loss += loss.item()
            # print("Running loss: {}".format(running_loss))
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(n_mod.cpu().data.numpy(), n_prob_label, list_metrics=["accuracy", "loss"])

            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(iter + 1,
                                                                                         num_iter_per_epoch,
                                                                                         epoch + 1, num_epochs,
                                                                                         training_metrics["loss"],
                                                                                         training_metrics[
                                                                                             "accuracy"]))
        average_loss = running_loss/num_iter_per_epoch
        print("Average loss: {}".format(average_loss))

            # print("pred during training: {}".format(np.argmax(prediction.cpu().data.numpy(), -1)))
        # evaluation of validation data
        model.eval()
        with torch.no_grad():

            validation_true = []
            validation_prob = []

            for batch in zip(x_val,y_val):
                _, n_mod = batch
                batch = [Variable(record).cuda() for record in batch]
                # get inputs
                t_iq, _ = batch
                # perform blind source separation
                # x = t_iq.view(-1, t_iq.shape[1] * t_iq.shape[2])
                # input = ica.fit_transform(x.cpu())
                # input = torch.Tensor(input).cuda()
                # input = input.view(-1, 128, 2)
                # forward pass
                t_predicted_label = model(t_iq)
                # using sigmoid to predict the label
                # t_predicted_label = F.sigmoid(t_predicted_label)

                validation_prob.append(t_predicted_label)
                validation_true.extend(n_mod.cpu().data.numpy())

                # print(validation_true, validation_prob)


            validation_prob = torch.cat(validation_prob, 0)
            validation_prob = validation_prob.cpu().data.numpy()
            # y_pred = np.argmax(validation_prob, -1)
            # print("val predicted:{}".format(validation_prob[0]))
            # print("val cleaned:{}".format(y_pred))

            # back to default:train
        model.train()

        test_metrics = get_evaluation(validation_true, validation_prob,
                                      list_metrics=["accuracy", "loss", "confusion_matrix"])

        output_file.write(
            "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {}"
            "\nAverage Loss: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, num_epochs,
                training_metrics["loss"],
                training_metrics["accuracy"],
                test_metrics["loss"],
                test_metrics["accuracy"],
                average_loss,
                test_metrics["confusion_matrix"]))
        print("\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, num_epochs, test_metrics["loss"],
                                                                    test_metrics["accuracy"]))

        # acc to the paper; half lr after 3 epochs
        if (num_epochs > 0 and num_epochs % 9 == 0):
            learning_rate = learning_rate / 10

        # saving the model with best accuracy
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(model, data_path+"tl_vsg_15_20_model")

    print("Training complete")
    print("-------------------------------------------")
    # print("Starting inference module")
    # inf.inference(data_path,x_test,y_test,raw_lables,snr_gen,"trained_cnn_intf_bpsk_snr20_sir25")


def get_evaluation(y_true, y_prob, list_metrics):
    # print(y_true)
    # print(y_prob)
    # print(type(y_true))
    # print(type(y_prob))
    y_pred = np.argmax(y_prob, -1)
    # print(y_pred)
    y_true = np.argmax(y_true,-1)
    # print(y_true)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


if __name__=="__main__":
    # path = "/media/backup/Arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/"
    path = "/media/backup/Arsenal/rf_dataset_inets/"
    train(path,30)
    # x = torch.tensor([[[1.6946e-01, -4.8737e-01],
    #                    [-1.7219e-02, -9.9859e-01],
    #                    [-1.8434e-01, -1.2288e+00],
    #                    [1.1188e-01, -7.9144e-01],
    #                    [-7.2184e-01, -1.1227e+00],
    #                    [-1.2231e+00, -1.3255e+00]],
    #
    #                   [[-1.5068e-01, 3.7411e-02],
    #                    [-9.9978e-02, 4.0878e-01],
    #                    [1.7568e-01, 1.2573e+00],
    #                    [-1.5838e-03, -2.3893e+00],
    #                    [1.0081e+00, -1.6239e+00],
    #                    [9.6221e-01, -3.4776e-01]]],
    #                  device='cuda:0')
    # x = x.view(-1,x.shape[1]*x.shape[2])
    # print(x.shape)
    # print(x)
    #
    # ica = FastICA(n_components=8,tol=1e-5,max_iter=1000)
    # out = ica.fit_transform(x.cpu())
    # print(out)
    # tList = [ica.fit_transform(sample.cpu()) for sample in torch.unbind(x, dim=0)]
    # out = torch.Tensor(tList).cuda()
    #
    # print(out)
    # signal=[]
    # for src in tList[0]:
    #     real = src.T[0]
    #     img = src.T[1]
    #     signal.append((real, img))
    # print(signal)
    # res = torch.stack(torch.Tensor(out), dim=0)
    # print(res.cuda())





