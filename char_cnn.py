import torch
import torch.nn as nn
from pytorch_model_summary import summary

class CharCNN(nn.Module):
    # input dim is 2 because of iq samples
    # sample size is 1024
    def __init__(self, n_classes, input_dim=2
                 , max_seq_length=1024, filters=256, kernel_sizes=[7, 7, 3, 3, 3, 3], pool_size=3,
                 n_fc_neurons=1024):
        super(CharCNN, self).__init__()

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
            nn.Conv1d(input_dim, filters, kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # layer 3,4,5
        self.conv3 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[2]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[3]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[4]),
            nn.ReLU()
        )

        # layer 6
        self.conv6 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[5]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        dimension = int(((max_seq_length - 96) / 27 * filters) - 94)

        # layer 7
        self.fc1 = nn.Sequential(
            nn.Linear(dimension, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # layer 8
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # layer 9
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        if filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        print(input)
        # print(input.shape)
        # changing dimensions to suit my network
        input = input.permute(0, 2, 1)
        output = self.conv1(input)
        # print(output.size())
        output = self.conv2(output)
        # print(output.size())
        output = self.conv3(output)
        # print(output.size())
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        # output = self.softmax(output)
        # print("final output:", output)

        return output


if __name__=="__main__":
    model = CharCNN(n_classes=24)
    print(summary(model,torch.ones((1,1024,2)),show_input=False, show_hierarchical=False))