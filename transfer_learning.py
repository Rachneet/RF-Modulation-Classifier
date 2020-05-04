import torch
from train import *
from cnn_model import *


def transfer_learning(path):
    model = torch.load(path+"tl_vsg_10_15_model",map_location='cuda:0')

    # check params of model
    # for i,param in model.named_parameters():
    #     print(i,param)

    freeze_layers = True

    # freezing all layers
    if freeze_layers:
        for i, param in model.named_parameters():
            param.requires_grad = False

    # freezing layers till conv layer 6
    ct = []
    for name, child in model.named_children():  # accessing layer names via named_children()
        # print(name,child)
        if "conv6" in ct:  # when conv6 is in list, make grad_true for further layers
            for params in child.parameters():
                params.requires_grad = True

        ct.append(name)

    # print("=================================")

    # view the freezed layers
    # for name, child in model.named_children():
    #     for name_2, params in child.named_parameters():
    #         print(name_2, params.requires_grad)

    # train the modified model
    train(path,model,num_epochs=10)


if __name__=="__main__":
    path = "/media/backup/Arsenal/thesis_results/"
    transfer_learning(path)