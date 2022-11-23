# import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import random
# from livelossplot import PlotLosses
import wandb
# from pycm import *

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, ToPILImage


class MNISTTensorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (Tensor): A tensor containing the data e.g. images
            targets (Tensor): A tensor containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        sample = (sample - 0.1307) / 0.3081
        sample = sample.view(1, 28, 28).float() / 255.
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class SimpleNet(nn.Module):
    def __init__(self, I=28 * 28, H=25, O=10):
        super(SimpleNet, self).__init__()
        self.input = nn.Linear(I, H)
        self.hidden = nn.Linear(H, O)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input(x.view(-1, 28 * 28)))
        x = self.hidden(x)
        return x


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # uses the inbuilt cudnn auto-tuner to find the fastest convolution
    # algorithms. -
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def set_device():
    global device
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
        print(
            "Cuda installed! Running on GPU %s!" %
            torch.cuda.get_device_name())

    else:
        print("No GPU available! Running on CPU")
    return True


def test_model_output_size(model, classes=10):
    x = torch.randn((1, 1, 28, 28)).to(device)
    y = model(x)
    assert(y.shape == (1, classes))


def make_MNIST_dataset(train, config, download=True):
    mnist = MNIST("./", download=download, train=train)

    if train:
        shuffler = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.valid_split,
            random_state=42).split(
            mnist.data,
            mnist.targets)
        train_idx, valid_idx = [(train_idx, validation_idx)
                                for train_idx, validation_idx in shuffler][0]

        X_train, y_train = mnist.data[train_idx], mnist.targets[train_idx]
        X_valid, y_valid = mnist.data[valid_idx], mnist.targets[valid_idx]

        trainds = MNISTTensorDataset(
            X_train, y_train.long(), config.train_transform)
        validds = MNISTTensorDataset(
            X_valid, y_valid.long(), config.test_transform)

        return trainds, validds

    else:
        X_test, y_test = mnist.data, mnist.targets
        testds = MNISTTensorDataset(
            X_test, y_test.long(), config.test_transform)
        return testds


def make_dataloader(
        dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)
    return loader


def set_up(config):
    # Get data and make datasets
    trainds, validds = make_MNIST_dataset(
        train=True, config=config, download=True)
    testds = make_MNIST_dataset(train=False, config=config, download=True)

    # Get data loaders
    train_loader = make_dataloader(trainds, config.batch_size)
    valid_loader = make_dataloader(validds, config.test_batch_size)
    test_loader = make_dataloader(testds, config.test_batch_size)

    # Make model
    try:
        model = globals()[config.model_name]

        model = model().to(device)
    except BaseException:
        raise NotImplementedError(
            "Model of name %s has not been found in this file" %
            config.model_name)
    config.model = model
    test_model_output_size(model, classes=10)

    # Make optimizer
    try:
        optimizer = getattr(torch.optim, config.optimizer_name)
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)

    except BaseException:
        raise NotImplementedError(
            "Optimizer of name %s has not been found in torch.optim" %
            config.optimizer_name)
    try:
        for g in optimizer.param_groups:
            g['momentum'] = config.momentum
    except BaseException:
        config.momentum = 0
        pass
    config.optimizer = optimizer

    # Make loss
    try:
        criterion = getattr(torch.nn, config.criterion_name)
        criterion = criterion()
    except BaseException:
        raise NotImplementedError(
            "Criterion of name %s has not been found in torch.nn" %
            config.criterion_name)
    config.criterion = criterion

    return model, criterion, optimizer, train_loader, valid_loader, test_loader


def train(model, optimizer, criterion, data_loader):
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        a2 = model(X.view(-1, 1, 28, 28))
        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss * X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(),
                                         y_pred.detach().cpu().numpy()) * X.size(0)
        optimizer.step()

    return train_loss / \
        len(data_loader.dataset), train_accuracy / len(data_loader.dataset)


def validate(model, criterion, data_loader):
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 1, 28, 28))
            loss = criterion(a2, y)
            validation_loss += loss * X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(
                y.cpu().numpy(), y_pred.cpu().numpy()) * X.size(0)

    return validation_loss / len(
        data_loader.dataset), validation_accuracy / len(data_loader.dataset)


def evaluate(model, data_loader):
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 1, 28, 28))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0), np.concatenate(ys, 0)


if __name__ == "__main__":

    # Set training hyperparameters
    hyperparameters = dict(
        epochs=100,
        classes=10,
        batch_size=128,
        test_batch_size=1000,
        learning_rate=1e-1,
        dataset="MNIST",
        model_name="SimpleNet",
        criterion_name="CrossEntropyLoss",
        optimizer_name="SGD",
        valid_split=0.2,
        train_transform=None,
        test_transform=None,
        momentum=0.5)

    # tell wandb to get started
    with wandb.init(config=hyperparameters, entity="dekape", project="pytorch-demo"):
        # Set seed and devices
        set_seed(42)
        set_device()

        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Set up run with hypeparameters
        model, criterion, optimizer, train_loader, valid_loader, test_loader = set_up(
            config)

        # Let wandb watch the model and the criterion
        wandb.watch(model, criterion)

        for epoch in range(config.epochs):
            train_loss, train_accuracy = train(
                model, optimizer, criterion, train_loader)
            validation_loss, validation_accuracy = validate(
                model, criterion, valid_loader)
            log = {
                "epoch": epoch + 1,
                "train_loss": train_loss.item(),
                "train_accuracy": train_accuracy.item(),
                "valid_loss": validation_loss.item(),
                "valid_accuracy": validation_accuracy.item()}
            print(log)
            wandb.log(log)
