import torch.nn as nn
import torch
from pytorch_model_summary import summary


class DNN(nn.Module):

    def __init__(self, input_dims, n_classes):
        super(DNN, self).__init__()

        self.fc1 = nn.Sequential(
        nn.Linear(input_dims, 4096),
        nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc8 = nn.Linear(64, n_classes)

    def forward(self, input):

        output = input.permute(0,2,1)
        output = output.reshape(*output.shape[:1], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.fc7(output)
        output = self.fc8(output)

        return output


if __name__=="__main__":
    model = DNN(2048, n_classes=8)
    print(summary(model,torch.ones((2,1024,2)),show_input=False, show_hierarchical=False))