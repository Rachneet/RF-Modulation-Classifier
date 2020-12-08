import torch.nn as nn
import numpy as np
import h5py as h5


class CNN(nn.Module):
    def __init__(self, n_classes, input_dim=2
                 ,max_seq_length=1024,filters=64,kernel_sizes=[3,3,3,3,3,3],pool_size=3,
                n_fc_neurons=128):

        super(CNN,self).__init__()

        self.filters = filters
        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.pool_size = pool_size

        # pooling in layer 1,2,6 ; pool =3
        # layers 7,8 and 9 are fully connected
        # 2 dropout modules between 3 fully connected layers
        # dropout - prevents overfitting in neural networks ; drops neurons with a certain probability
        # (from 2014 paper Dropout by Srivastava et al.); only during training

        # layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, filters, kernel_sizes[0],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        # layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_sizes[1],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        # layer 3,4,5
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_sizes[2],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_sizes[3],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_sizes[4],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        # layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_sizes[5],padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        # dimension = int(((max_seq_length - 96) / 27 * filters)-94)
        # print("Dimension before FC layer: ", dimension)

        # layer 7
        self.fc1 = nn.Sequential(
            nn.Linear(n_fc_neurons,n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # layer 8
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # batch norm
        # self.batchNorm = nn.BatchNorm1d(n_fc_neurons)

        # layer 9
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):

        input = input.permute(0,2,1)
        input = input.unsqueeze(dim=3)
        output = self.conv1(input)
        # print(output.size())
        output = self.conv2(output)
        # print(output.size())
        output = self.conv3(output)
        # print(output.size())
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        # print("output after conv6:", output.shape)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output



if __name__=="__main__":

    path = "/media/backup/Arsenal/rf_dataset_inets/feature_set_training_fc8_vsg20.h5"
    data = h5.File(path,'r')
    # features,pred_labels,true_labels = data['features'],data['pred_labels'],data['true_labels']
    # data = np.load(path, allow_pickle=True)
    labels = [0,1,2,3,4,5,6,7]
    # print(data.files)
    iq = data['features']
    t_labels = data['true_labels']
    pred_labels = data['pred_labels']

    agg_features=[]
    for i in range(len(labels)):
        features = np.zeros(8, )
        count = 0          # reset feature vector and count
        for j in range(iq.shape[0]):
            if pred_labels[j] == t_labels[j]:    # get only the correct features
                if pred_labels[j] == labels[i]:
                    features += iq[j]
                    count += 1
            else:
                continue
        agg_features.append(features/count)

    print(np.array(agg_features).shape)
    np.savez("/media/backup/Arsenal/rf_dataset_inets/features_fc8_mav.npz",
             features=np.array(agg_features))
