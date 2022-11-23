import numpy as np
import random
import torch
import torch.nn as nn

__all__ = ['set_seed', "train_lstm_mse", "validate_mse",
           "train_lstm_SSIM", "validate_SSIM", "evaluate"]

# specify the device
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

# set random seed


def set_seed(seed):
    """
    Set a random seed for training

    Parameters
    ----------
    seed : int

    Returns
    ----------
    None

    Examples
    --------
    >>> import tools.fc_lstm as fclstm
    >>> fclstm.set_seed(42)
    True
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

# LSTM model
# Here we specify the input and output size as customised parameters


class LSTM(nn.Module):
    """
    Initialize the Long short term memory network for training later.

    Parameters
    ----------
    input_size : float/int
    hidden_layer_size : float/int
    output_size : float/int

    Notes
    -----
    see
    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
    for further details
    """

    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, x):
        # batch_size = x.size(0)
        lstm_out, self.hidden_cell = self.lstm(
            x.view(len(x), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions[-1]

# MSE as loss function


def train_lstm_mse(model, optimizer, criterion, data_loader):
    """
    Train the model with MSE loss function

    Parameters
    ----------
    model: nn.Module
    optimizer: torch.optim.*
    criterion: torch.nn.MSELoss()
    dataLoader: torch.utils.data.DataLoader

    Returns
    ----------
    The total training loss after training (float)

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> criterion = nn.MSELoss()
    >>> result = fclstm.train_lstm_mse(model, optimizer, criterion, dataloader)
    >>> result != 0
    tensor(True)
    """
    model.train()
    train_loss = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(
                1, 1, model.hidden_layer_size).to(device), torch.zeros(
                1, 1, model.hidden_layer_size).to(device))
        y_pred = model(X[0])
        loss = criterion(y_pred, y.squeeze())
        loss.backward()
        train_loss += loss * X.size(0)
        optimizer.step()

    return train_loss / len(data_loader)


def validate_mse(model, criterion, data_loader):
    """
    Validate the trained model with MSE loss function

    Parameters
    ----------
    model: nn.Module
    criterion: torch.nn.MSELoss()
    dataLoader: torch.utils.data.DataLoader

    Returns
    ----------
    The total validation loss (float)

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> criterion = nn.MSELoss()
    >>> result = fclstm.validate_mse(model, criterion, dataloader)
    >>> result != 0
    tensor(True)
    """
    model.eval()
    validation_loss = 0
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_pred = model(X[0])
            loss = criterion(y_pred, y.squeeze())
            validation_loss += loss * X.size(0)
    return validation_loss / len(data_loader.dataset)

# functions with SSIM


def train_lstm_SSIM(model, optimizer, criterion, data_loader):
    """
    Train the model with SSIM loss function

    Parameters
    ----------
    model: nn.Module
    optimizer: torch.optim.*
    criterion: torch.nn.MSELoss()
    dataLoader: torch.utils.data.DataLoader

    Returns
    ----------
    The total training loss after training (float)

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> from pytorch_msssim import ssim
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> criterion = ssim
    >>> result = fclstm.train_lstm_SSIM(model, optimizer, criterion, dataloader)
    >>> result != 0
    tensor(True)
    """
    model.train()
    train_loss = 0
    for X, y in data_loader:
        img_size = np.sqrt(X.size(-1)).astype(int)
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(
                1, 1, model.hidden_layer_size).to(device), torch.zeros(
                1, 1, model.hidden_layer_size).to(device))
        y_pred = model(X[0])
        loss = 1 - criterion(y_pred.reshape(1, 1, img_size, img_size),
                             y.reshape(1, 1, img_size, img_size))
        loss.backward()
        train_loss += loss * X.size(0)
        optimizer.step()

    return train_loss / len(data_loader)


def validate_SSIM(model, criterion, data_loader):
    """
    Validate the model with SSIM loss function

    Parameters
    ----------
    model: nn.Module
    criterion: torch.nn.MSELoss()
    dataLoader: torch.utils.data.DataLoader

    Returns
    ----------
    The total validation loss (float)

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> from pytorch_msssim import ssim
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> criterion = ssim
    >>> result = fclstm.validate_SSIM(model, criterion, dataloader)
    >>> result != 0
    tensor(True)
    """
    model.eval()
    validation_loss = 0
    for X, y in data_loader:
        img_size = np.sqrt(X.size(-1)).astype(int)
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_pred = model(X[0])
            loss = 1 - criterion(y_pred.reshape(1, 1, img_size,
                                                img_size), y.reshape(1, 1, img_size, img_size))
            validation_loss += loss * X.size(0)
    return validation_loss / len(data_loader.dataset)


def evaluate(model, data_loader):
    """
    Evaluate the model on test data_loader

    Parameters
    ----------
    model: nn.Module
    dataLoader: torch.utils.data.DataLoader

    Returns
    ----------
    The expected resultant output of model and ground truth image

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> result_1,result_2 = fclstm.evaluate(model,dataloader)
    >>> result_1.size != 0
    True
    >>> result_2.size != 0
    True
    """
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_pred = model(X[0])
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
    return np.concatenate(y_preds, 0), np.concatenate(ys, 0)

# These functions are not compulsory but we can use them in Gridsearch


def train_mse(seed, epochs, batch_size, model, criterion, optimizer,
              train_loader, val_loader, test_loader):
    """
    Train epochs with MSE loss function

    Parameters
    ----------
    seed: random seed (int)
    epochs: int
    batch_size: int
    model: nn.Module
    criterion: nn.MSELoss()
    optimizer: torch.optim.*
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    Returns
    ----------
    bool

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> seed = 42
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> epochs = 10
    >>> batch_size = 1
    >>> criterion = nn.MSELoss()
    >>> fclstm.train_mse(seed, epochs, batch_size, model, criterion, optimizer,dataloader, dataloader, dataloader)
    True
    """
    set_seed(seed)
    epochs = epochs
    batch_size = batch_size
    model = model
    criterion = criterion
    optimizer = optimizer
    train_loader = train_loader
    val_loader = val_loader
    test_loader = test_loader
    for _ in range(epochs):
        train_lstm_mse(model, optimizer, criterion, train_loader)
    return True


def train_ssim(seed, epochs, batch_size, model, criterion, optimizer,
               train_loader, val_loader, test_loader):
    """
    Train epochs with SSIM loss function

    Parameters
    ----------
    seed: random seed (int)
    epochs: int
    batch_size: int
    model: nn.Module
    criterion: pytorch_msssim.ssim
    optimizer: torch.optim.*
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    Returns
    ----------
    bool

    Examples
    --------
    >>> import torch
    >>> import tools.fc_lstm as fclstm
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> from pytorch_msssim import ssim
    >>> train = torch.stack([torch.randn((1,128*128))]*10)
    >>> test = torch.stack([torch.randn((128*128))]*10)
    >>> data = TensorDataset(train,test)
    >>> dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    >>> seed = 42
    >>> model = fclstm.LSTM(128*128,150,128*128)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> epochs = 10
    >>> batch_size = 1
    >>> criterion = ssim
    >>> fclstm.train_ssim(seed, epochs, batch_size, model, criterion, optimizer,dataloader, dataloader, dataloader)
    True
    """
    set_seed(seed)
    epochs = epochs
    batch_size = batch_size
    model_ssim = model
    criterion = criterion
    optimizer = optimizer
    train_loader = train_loader
    val_loader = val_loader
    test_loader = test_loader
    for _ in range(epochs):
        train_lstm_SSIM(model_ssim, optimizer, criterion, train_loader)
    return True

# export KMP_DUPLICATE_LIB_OK=TRUE


if __name__ == "__main__":
    import doctest
    doctest.testmod()
