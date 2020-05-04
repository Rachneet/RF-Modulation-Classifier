import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as tf


def image_dataloader(path,batch_size):

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 4}

    transform = tf.Compose(
        [tf.Resize(400),tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.ImageFolder(
        root=path+"train",
        transform=transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=path + "test",
        transform=transform
    )

    train_set = DataLoader(train_dataset,**training_params)
    test_set = DataLoader(test_dataset, **training_params)

    return train_set,test_set

    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)



if __name__=="__main__":
    image_dataloader('/media/backup/Arsenal/rf_dataset_inets/image_dataset/',128)