"""Transfer Learning for RF Signals.
This illustrates how one could fine-tune a pre-trained
network (by default, a custom CNN is used) using pytorch-lightning. The dataset is
trained for 15 epochs. The training consists in three stages. From epoch 0 to
4, the feature extractor (the pre-trained network) is frozen except maybe for
the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained
as a single parameters group with lr = 1e-2. From epoch 5 to 9, the last two
layer groups of the pre-trained network are unfrozen(here: the conv layers) and added to the
optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3 for the
first parameter group in the optimizer). Eventually, from epoch 10, all the
remaining layer groups of the pre-trained network are unfrozen and added to
the optimizer as a third parameter group. From epoch 10, the parameters of the
pre-trained network are trained with lr = 1e-5 while those of the classifier
are trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch
import torch.nn as nn
from typing import Optional, Generator, Union
from torch.optim import Optimizer
import pytorch_lightning as pl
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import math
from sklearn import metrics
import os
import csv
from torch.utils.data import DataLoader,SubsetRandomSampler
from pytorch_lightning.logging.neptune import NeptuneLogger

from py_lightning import LightningCNN, DatasetFromHDF5
from cnn_model import CNN

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
# --------------------------------------------Utility Functions--------------------------------------------------


def _make_trainable(module:nn.Module):
    """
     Unfreezes a given module
    """
    for params in module.parameters():
        params.requires_grad = True
    module.train()


def _recursive_freeze(module:nn.Module, train_bn: bool = True):
    """
         Freezes layers a given module
    """
    children = list(module.children())   # goes inside sequential and checks
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            # freeze all except BN layers
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:      # if we have no sequential models stacked
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module:nn.Module, n:Optional[int]=None, train_bn:bool=True):
    """
    freezes the layers upto n index
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: torch.nn.Module,
                  train_bn: bool = True):
    """
    Yields the trainable parameters of a given module.
    Returns: Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: torch.nn.Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


#  --------------------------------------------- Pytorch-lightning module ------------------------------------------

class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained CNN.
       Args:
           hparams: Model hyperparameters
           data_path: Path where the data will be
       """

    def __init__(self,
                 hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.all_true = []
        self.all_pred = []
        self.all_snr = []
        self.data_path = hparams.data_path
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""
        # 1. Load pre-trained model
        backbone = self.hparams.backbone

        _layers = list(backbone.children())[:-3]  # all except fc layers  # maybe modify later
        self.feature_extractor = nn.Sequential(*_layers)
        freeze(module=self.feature_extractor,train_bn=self.hparams.train_bn)  # freeze all layers

        # 2. Classifier
        _fc_layers = [nn.Linear(128, 64),
                      nn.Linear(64, 32),
                      nn.Linear(32, self.hparams.n_classes)]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss function
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = x.unsqueeze(dim=3)
        # 1. Feature extraction
        x = self.feature_extractor(x)
        # print(x.shape)
        x = x.squeeze(2)
        x = x.reshape(*x.shape[:1], -1)

        # modify later
        x = self.fc(x)
        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.hparams.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,train_bn=self.hparams.train_bn)

        elif self.hparams.milestones[0] <= epoch < self.hparams.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.hparams.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.hparams.milestones[0]:  # unfreeze last 2
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                                          optimizer=optimizer,train_bn=self.hparams.train_bn)

        elif self.current_epoch == self.hparams.milestones[1]:    # unfreeze all except last 2
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                                          optimizer=optimizer,train_bn=self.hparams.train_bn)

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y, _ = batch
        y_logits = self.forward(x)
        y_true = torch.max(y, 1)[1]

        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_logits, y_true)
        num_correct = torch.eq(torch.max(y_logits,1)[1], y_true).sum()
        # accuracy = num_correct.detach().cpu().data.numpy()/list(y_true.size())[0]

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'train_loss': train_loss_mean,
                        'train_acc': train_acc_mean,
                        'step': self.current_epoch}}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y, _ = batch
        y_logits = self.forward(x)
        y_true = torch.max(y, 1)[1]

        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_logits, y_true)
        num_correct = torch.eq(torch.max(y_logits, 1)[1], y_true).sum()

        return {'val_loss': val_loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean,
                        'step': self.current_epoch}}

    def test_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y, z = batch
        y_logits = self.forward(x)
        y_true = torch.max(y, 1)[1]
        y_hat = torch.max(y_logits, 1)[1]
        # 2. Compute loss & accuracy:
        test_loss = self.loss(y_logits, y_true)
        num_correct = torch.eq(y_hat, y_true).sum()

        return {'test_loss': test_loss,
                'num_correct': num_correct,
                'true_label': y_true,
                'pred_label': y_hat,
                'snrs' : z}

    def test_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        test_loss_mean = torch.stack([output['test_loss']
                                     for output in outputs]).mean()
        test_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        test_loss_mean /= (len(outputs) * self.hparams.batch_size)
        for output in outputs:
            self.all_true.extend(output['true_label'].detach().cpu().data.numpy())
            self.all_pred.extend(output['pred_label'].detach().cpu().data.numpy())
            self.all_snr.extend(output['snrs'].detach().cpu().data.numpy())

        accuracy = metrics.accuracy_score(self.all_true, self.all_pred)
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
        neptune_logger.experiment.log_metric('test_accuracy(true)', accuracy)
        neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR + "output.csv")
        return {'log': {'test_loss': test_loss_mean,
                        'test_acc': test_acc_mean,
                        'step': self.current_epoch}}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                                    lr=self.hparams.lr,momentum=self.hparams.momentum)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=self.hparams.milestones,
                                gamma=self.hparams.lr_scheduler_gamma)

        return [optimizer], [scheduler]

    def prepare_data(self, valid_fraction=0.05, test_fraction=0.2):
        dataset = DatasetFromHDF5(self.data_path, 'iq', 'labels', 'snrs')
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
                                   shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers,
                                   sampler=train_sampler)
        self.val_dataset = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                      shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers,
                                      sampler=valid_sampler)
        self.test_dataset = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                       shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers,
                                       sampler=test_sampler)


    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataset

    @pl.data_loader
    def val_dataloader(self):
        return self.val_dataset

    @pl.data_loader
    def test_dataloader(self):
        return self.test_dataset

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser])
        # model = torch.load("/home/rachneet/thesis_results/vsg_vier_mod/epoch=15.ckpt",map_location='cuda:0')

        model = LightningCNN.load_from_checkpoint(
            checkpoint_path='/home/rachneet/thesis_results/vsg_vier_mod/epoch=15.ckpt',
            map_location=None
        )
        parser.add_argument('--backbone', default=model)
        parser.add_argument('--epochs',default=15,type=int)
        parser.add_argument('--batch-size',default=512,type=int)
        parser.add_argument('--shuffle', default=False, type=bool)
        parser.add_argument('--gpus', default=0, type=int)
        parser.add_argument('--lr','--learning-rate',default=1e-2,type=float)
        parser.add_argument('--momentum', default=0.9)
        parser.add_argument('--lr-scheduler-gamma',default=1e-1,type=float)
        parser.add_argument('--num-workers',default=10,type=int)
        parser.add_argument('--train-bn',default=True,type=bool)
        parser.add_argument('--milestones',default=[5, 10],type=list)
        parser.add_argument('--max_epochs', default=15)
        # adding model params
        parser.add_argument('--in_dims', default=2)
        parser.add_argument('--filters', default=64)
        parser.add_argument('--kernel_size', type=list, nargs='+', default=[3, 3, 3, 3, 3, 3])
        parser.add_argument('--pool_size', default=3)
        parser.add_argument('--fc_neurons', default=128)
        parser.add_argument('--n_classes', default=4)

        return parser

# =========================================NEPTUNE AI===============================================================


CHECKPOINTS_DIR = '/home/rachneet/thesis_results/tl_vsg_deepsig_new/'           # change this
neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmU"
            "uYWkiLCJhcGlfa2V5IjoiZjAzY2IwZjMtYzU3MS00ZmVhLWIzNmItM2QzOTY2NTIzOWNhIn0=",
    project_name="rachneet/sandbox",
    experiment_name="tl_vsg_deepsig_new",   # change this  for new runs
)

# ===================================================================================================================

def test_lightning(hparams: argparse.Namespace):
    model = TransferLearningModel.load_from_checkpoint(
        CHECKPOINTS_DIR + 'epoch=14.ckpt'
    )

    # exp = Experiment(name='test_vsg20',save_dir=os.getcwd())
    # logger = TestTubeLogger('tb_logs', name='CNN')
    # callback = [test_callback()]
    # print(neptune_logger.experiment.name)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)
    trainer = pl.Trainer(logger=neptune_logger, gpus=hparams.gpus, checkpoint_callback=model_checkpoint)
    dataset = DatasetFromHDF5(hparams.data_path, 'iq', 'labels', 'snrs')
    num_train = len(dataset)
    indices = list(range(num_train))
    val_split = int(math.floor(0.05 * num_train))
    test_split = val_split + int(math.floor(0.2 * num_train))
    training_params = {"batch_size": hparams.batch_size, "num_workers": hparams.num_workers}

    if not ('shuffle' in training_params and not training_params['shuffle']):
        np.random.seed(4)
        np.random.shuffle(indices)
    if 'num_workers' not in training_params:
        training_params['num_workers'] = 1

    train_idx, valid_idx, test_idx = indices[test_split:], indices[:val_split], indices[val_split:test_split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    test_dataset = DataLoader(dataset, batch_size=hparams.batch_size,
                              shuffle=hparams.shuffle, num_workers=hparams.num_workers, sampler=test_sampler)
    trainer.test(model, test_dataloaders=test_dataset)
    # Save checkpoints folder
    neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
    # You can stop the experiment
    neptune_logger.experiment.stop()


def main(hparams: argparse.Namespace) -> None:
    """Train the model.
    Args:
        hparams: Model hyper-parameters
    """
    # print(vars(hparams))
    model = TransferLearningModel(hparams)

    if not os.path.exists(CHECKPOINTS_DIR):   # redundant maybe
        os.makedirs(CHECKPOINTS_DIR)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(logger=neptune_logger,
        gpus=hparams.gpus,
        max_nb_epochs=hparams.max_epochs,
        checkpoint_callback=model_checkpoint)

    # trainer.fit(model)
    # # load best model for testing
    # file_name = ''
    # for subdir, dirs, files in os.walk(CHECKPOINTS_DIR):
    #     for file in files:
    #         if file[-4:] == 'ckpt':
    #             file_name = file

    model = TransferLearningModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS_DIR+"epoch=14.ckpt",
        # hparams=hparams,
        # map_location="cuda:0"
    )

    trainer.test(model)
    neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
    neptune_logger.experiment.stop()


def get_args() -> argparse.Namespace:
    root_path = "/home/rachneet/rf_dataset_inets/dataset_deepsig_vier_new.hdf5"
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path',default=root_path, help='path to dataset')
    parser = TransferLearningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()

# --------------------------------------------------MAIN-------------------------------------------------------------
# def load_weights(self,hparams):
#     checkpoint_path = '/media/backup/Arsenal/thesis_results/lightning_vsg_snr_0/version_SAN-10/checkpoints' \
#                       'epoch=24.ckpt'
#     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, )
#     pretrained_dict = checkpoint["state_dict"]
#     model = TransferLearningModel(hparams, data_path=hparams.data_path)
#     model_dict = self.state_dict()

if __name__ == "__main__":
    # model = CNN(n_classes=8)
    # _recursive_freeze(model)
    # x = [1,2,3,4,5,6]
    # print(x[-2:])
    # print(x[:-2])
    # x = torch.tensor([[1,2,3,1],[1,2,3,4]])
    # print(x.shape)
    # x = x.flatten()
    # print(x.shape)
    # true = torch.tensor([1,2,3,0])
    # print(torch.eq(pred,true).sum().detach().cpu().data.numpy())
    # correct = (torch.eq(pred,true).sum().detach().cpu().data.numpy()/list(pred.size())[0])
    # print(torch.tensor(correct))
    main(get_args())
    # test_lightning(get_args())
    # obj = TransferLearningModel()
    # pass
