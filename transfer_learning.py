import torch
from train import *
from cnn_model import *
from py_lightning import LightningCNN


def transfer_learning(path):
    model = LightningCNN.load_from_checkpoint(
        checkpoint_path='/home/rachneet/thesis_results/vsg_vier_mod/epoch=15.ckpt',
        map_location=None
    )
    # model = torch.load(path+"vsg_vier_mod/epoch=15.ckpt",map_location='cuda:0')

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
    train(path+"tl_vsg_deepsig/",model,num_epochs=10)


if __name__=="__main__":
    path = "/home/rachneet/thesis_results/"
    transfer_learning(path)