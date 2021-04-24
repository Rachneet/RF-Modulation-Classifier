from models.pytorch.resnet import resnet101
from data_processing.image_dataloader import image_dataloader

import torch.nn as nn
import torch
import numpy as np
from sklearn import metrics
from torch.autograd import Variable


def train(data_path,num_epochs):

    train_set, test_set = image_dataloader(data_path,44)  #GOLD_XYZ_OSC.0001_1024.hdf5
    print("Data loaded and batched...")
    #model = cnn_model.CNN(n_classes=8)
    model = resnet101(3,8)
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

    num_iter_per_epoch = len(train_set)

    best_accuracy = 0
    # reg_lambda = 0.1

    output_file = open("rf_resnet101_spectrogram_16k.txt", "w")
    # print(zip(x_train_gen, y_train_gen))

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        average_loss = 0

        for iter, batch in enumerate(train_set):

            _, n_mod = batch

            batch = [Variable(record).cuda() for record in batch]
            optimizer.zero_grad()
            t_iq, t_mod = batch
            prediction = model(t_iq)
            n_prob_label = prediction.cpu().data.numpy()

            loss = criterion(prediction, t_mod)   # + l2_reg*reg_lambda
            running_loss += loss.item()
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

        # evaluation of validation data
        model.eval()
        with torch.no_grad():

            validation_true = []
            validation_prob = []

            for batch in test_set:
                _, n_mod = batch

                # setting volatile to true because we are in inference mode
                # we will not be backpropagating here
                # conserving our memory by doing this
                # edit:volatile is deprecated now; using torch.no_grad();see above
                batch = [Variable(record).cuda() for record in batch]
                # i = Variable(i).cuda()
                # j = Variable(j).cuda()
                # get inputs
                t_iq, _ = batch
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
                "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nAverage Loss: {}  \nTest confusion matrix: \n{}\n\n".format(
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
            torch.save(model, "trained_resnet101_spectrogram_16k")


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    # print(y_pred)
    # y_true = np.argmax(y_true,-1)
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


if __name__ == "__main__":
    path = "/home/rachneet/datasets/spectrogram_dataset/"
    train(path,num_epochs=20)