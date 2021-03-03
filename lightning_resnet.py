import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, Subset, Dataset
from sklearn import preprocessing
import time
from torch.utils.data.sampler import SubsetRandomSampler
import h5py as h5
import math
import os
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from test_tube import Experiment
# from comet_ml import Experiment
# from pytorch_lightning.logging import CometLogger
# from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn import metrics
from collections import OrderedDict
import csv
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from argparse import Namespace
torch.manual_seed(4)  # for reproducibility of results

from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('always',ConvergenceWarning)

from resnet import *

# ================================================Visualization=============================================

# comet_logger = CometLogger(
#     api_key="gKzrti6C84TsoTTyqlT5OHarD",
#     workspace="rachneet", # Optional
#     project_name="Master_Thesis" # Optional
#     # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
#     # experiment_name="default" # Optional
# )

# ====================================================================================================================


class DatasetFromHDF5(Dataset):
    def __init__(self, filename, iq,labels,snrs):
        self.filename = filename
        self.iq = iq
        self.labels = labels
        self.snrs = snrs
        # self.data = preprocessing.scale(self.data, with_mean=False)

    def __len__(self):
        with h5.File(self.filename, 'r') as file:
            lens = len(file[self.labels])
        return lens

    def __getitem__(self, item):
        with h5.File(self.filename, 'r') as file:
            data = file[self.iq][item]
            label = file[self.labels][item]
            snr = file[self.snrs][item]
        # ----------- Blind source separation ------------------------
        # x = np.expand_dims(data, axis=0)
        # x = x.reshape(-1, 256)
        # S = compute_ica(x)
        # out = np.dot(S, x)
        # signals = out.reshape(-1, 128, 2)
        # -------------------------------------------------------------
        # data = preprocessing.scale(data,with_mean=False).astype(np.float32)
        data = data.astype(np.float32)
        label = label.astype(np.float32)
        snr = snr.astype(np.int8)
        return data,label,snr

# ===============================================MODEL==============================================================

class LightningResnet(pl.LightningModule):
    def __init__(self, hparams):

        super(LightningResnet,self).__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.all_true, self.all_pred, self.all_snr = [], [], []  # for final metrics calculation
        self.hparams = hparams
        # print(vars(hparams)
        # get model
        self.res_enc = ResnetEncoder(hparams.in_dims,hparams.block_sizes,hparams.depths,block=hparams.res_block)
        self.res_dec = ResnetDecoder(self.res_enc.blocks[-1].blocks[-1].expanded_channels, hparams.n_classes)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(dim=3)
        x = self.res_enc(x)
        x = self.res_dec(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,momentum=self.hparams.momentum)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)   # dynamic reduction based on val_loss
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,18,27],gamma=0.1)
        return [optimizer],[scheduler]

    def cross_entropy_loss(self,logits, labels):
        loss = nn.CrossEntropyLoss()
        return loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        logits = self.forward(x)
        y = torch.max(y, 1)[1]
        loss = self.cross_entropy_loss(logits,y)

        tqdm_dict = {'train_loss': loss}
        # comet_logger.experiment.log_model('train_loss', loss)
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output


    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        y_pred = self.forward(x)
        y = torch.max(y,1)[1]
        # print(y)
        loss = self.cross_entropy_loss(y_pred,y)

        # accuracy
        y_hat = torch.max(y_pred,1)[1]
        val_acc = torch.sum(y == y_hat).item() / len(y)
        val_acc = torch.tensor(val_acc)

        output = OrderedDict({
            'val_loss': loss,
            'val_acc': val_acc,
        })

        return output

    def validation_epoch_end(self, outputs):
        # called at the end of a validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:  # not implemented
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:   # not implemented
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': {'val_loss': val_loss_mean,'val_acc':val_acc_mean},
                  'val_loss': val_loss_mean}
        return result

    def test_step(self, batch, batch_idx):
        # to do
        x, y, z = batch
        y_pred = self.forward(x)
        y = torch.max(y,1)[1]
        loss = self.cross_entropy_loss(y_pred,y)

        # accuracy
        y_hat = torch.max(y_pred,1)[1]
        test_acc = torch.sum(y == y_hat).item() / len(y)
        test_acc = torch.tensor(test_acc)

        # if batch_idx == int(len(y)/self.hparams.batch_size):
        output = OrderedDict({
            'test_loss': loss,
            'test_acc': test_acc,
            'true_label': y,
            'pred_label': y_hat,
            'snrs' : z,
        })

        return output

    def test_epoch_end(self, outputs):
        # called at the end of a test epoch
        # outputs is an array with what you returned in test_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        # print(outputs)
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = output['test_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:  # not implemented
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

            # reduce manually when using dp
            test_acc = output['test_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:  # not implemented
                test_acc = torch.mean(test_acc)

            test_acc_mean += test_acc
            self.all_true.extend(output['true_label'].detach().cpu().data.numpy())
            self.all_pred.extend(output['pred_label'].detach().cpu().data.numpy())
            self.all_snr.extend(output['snrs'].detach().cpu().data.numpy())

        confusion_matrix = metrics.confusion_matrix(self.all_true,self.all_pred)
        accuracy = metrics.accuracy_score(self.all_true,self.all_pred)
        # save results in csv
        fieldnames = ['True_label', 'Predicted_label', 'SNR']

        if not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)

        with open(CHECKPOINTS_DIR + "output.csv", 'w', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for i, j, k in zip(self.all_true, self.all_pred, self.all_snr):
                writer.writerow(
                    {'True_label': i.item(), 'Predicted_label': j.item(), 'SNR': k})

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        tqdm_dict = OrderedDict({'Test_loss': test_loss_mean, 'Test_acc(mean)': test_acc_mean,
                                'True_accuracy': accuracy, 'Confusion_matrix': confusion_matrix})

        neptune_logger.experiment.log_metric('test_accuracy', accuracy)
        neptune_logger.experiment.log_metric('test_loss', test_loss_mean)
        # Log charts
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(self.all_true, self.all_pred, ax=ax)
        neptune_logger.experiment.log_image('confusion_matrix', fig)
        # Save checkpoints folder
        neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR + "output.csv")
        result = {'progress_bar': tqdm_dict, 'log': {'test_loss':test_loss_mean}}
        return result


    def prepare_data(self, valid_fraction=0.05, test_fraction=0.2):
        dataset = DatasetFromHDF5(self.hparams.data_path, 'iq', 'labels', 'snrs')
        num_train = len(dataset)
        indices = list(range(num_train))
        val_split = int(math.floor(valid_fraction * num_train))
        test_split = val_split + int(math.floor(test_fraction * num_train))
        training_params = {"batch_size": self.hparams.batch_size,
                           "num_workers": self.hparams.num_workers}

        if not ('shuffle' in training_params and not training_params['shuffle']):
            np.random.seed(4)
            np.random.shuffle(indices)
        if 'num_workers' not in training_params:
            training_params['num_workers'] = 1

        train_idx, valid_idx, test_idx = indices[test_split:], indices[:val_split], indices[val_split:test_split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        self.train_dataset = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                   shuffle=self.hparams.shuffle,num_workers=self.hparams.num_workers,
                                   sampler=train_sampler)
        self.val_dataset = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=self.hparams.shuffle,num_workers=self.hparams.num_workers,
                                 sampler=valid_sampler)
        self.test_dataset = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                  shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers,
                                  sampler=test_sampler)


    # @pl.data_loader
    def train_dataloader(self):
        return self.train_dataset

    # @pl.data_loader
    def val_dataloader(self):
        return self.val_dataset

    # @pl.data_loader
    def test_dataloader(self):
        return self.test_dataset

# ==================================================================================================================

# class test_callback(pl.Callback):
#
#     def on_test_end(self,trainer,output):
#         print("Test ended")
#         print(trainer)
#         print(output)
# ----------------------------------------Testing the model-----------------------------------------------

# function to test the model separately
def test_lightning(hparams):

    model = LightningResnet(hparams)
    checkpoint_path = '/media/backup/Arsenal/thesis_results/res_intf_bpsk_snr10_all/epoch=12.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # model = LightningResnet.load_from_checkpoint(
    # checkpoint_path='/media/backup/Arsenal/thesis_results/res_intf_free_usrp_all/epoch=4.ckpt',
    # hparams=hparams,
    # map_location='cuda:0'
    # )
    # exp = Experiment(name='test_vsg20',save_dir=os.getcwd())
    # logger = TestTubeLogger('tb_logs', name='CNN')
    # callback = [test_callback()]
    # print(neptune_logger.experiment.name)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)
    trainer = Trainer(logger=neptune_logger, gpus=hparams.gpus,checkpoint_callback=model_checkpoint)
    trainer.test(model)
    # Save checkpoints folder
    neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
    # You can stop the experiment
    neptune_logger.experiment.stop()

# -------------------------------------------------------------------------------------------------------------------
CHECKPOINTS_DIR = '/home/rachneet/thesis_results/res_vsg_vier_mod/'
neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmU"
            "uYWkiLCJhcGlfa2V5IjoiZjAzY2IwZjMtYzU3MS00ZmVhLWIzNmItM2QzOTY2NTIzOWNhIn0=",
    project_name="rachneet/sandbox",
    experiment_name="res_vsg_vier_mod",   # change this for new runs
)

# ---------------------------------------MAIN FUNCTION TRAINER-------------------------------------------------------

def main(hparams):

    model = LightningResnet(hparams)
    # exp = Experiment(save_dir=os.getcwd())
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(logger=neptune_logger,gpus=hparams.gpus,max_nb_epochs=hparams.max_epochs,
                      add_log_row_interval=100,log_save_interval=200, checkpoint_callback=model_checkpoint,
                      early_stop_callback=early_stop_callback)
    trainer.fit(model)
    # load best model
    file_name = ''
    for subdir, dirs, files in os.walk(CHECKPOINTS_DIR):
        for file in files:
            if file[-4:] == 'ckpt':
                file_name = file
    model = LightningResnet.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS_DIR+file_name,
        hparams=hparams,
        map_location=None
    )
    trainer.test(model)
    neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
    neptune_logger.experiment.stop()
    # note: here dataloader can also be passed to the .fit function

# ==================================================PASSING ARGS=====================================================


if __name__=="__main__":

    path = "/home/rachneet/rf_dataset_inets/dataset_vsg_vier_mod.h5"
    out_path = "/home/rachneet/thesis_results/"

    parser = ArgumentParser()
    parser.add_argument('--output_path', default=out_path)
    parser.add_argument('--data_path', default=path)
    parser.add_argument('--gpus', default=0)
    parser.add_argument('--max_epochs', default=30)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--num_workers', default=10)
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--learning_rate', default=1e-2)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--in_dims', default=2)
    parser.add_argument('--n_classes', default=4)
    parser.add_argument('--block_sizes',type=list, default=[64,128,256,512])
    parser.add_argument('--depths', type=list, default=[3, 4, 23, 3])
    parser.add_argument('--res_block', default=ResnetBottleneckBlock)

    args = parser.parse_args()

    main(args)
    # test_lightning(args)

