import tools.fc_lstm as fclstm
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from pytorch_msssim import ssim
import pytest

# specify the device to run
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

# Note: In this Pytest we randomly generate some tensors as dataset for
# dataloader

# dummy dataset
# Here we specify the input and output size as customised parameters
##########################################################################
train = []
for _ in range(10):
    train.append(torch.randn((1, 128 * 128)))

test = []
for _ in range(10):
    test.append(torch.randn((128 * 128)))

test = torch.stack(test)
train = torch.stack(train)

dummy_data = TensorDataset(train, test)
dummy_loader = DataLoader(dummy_data, batch_size=1,
                          shuffle=True, num_workers=0)
dummy_model = fclstm.LSTM(128 * 128, 150, 128 * 128)
dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
##########################################################################

# parameterize(param chars,[list of param tuple])
# the test parameters are based on the hypothesis that input images are 2D
# and output is 1D


@pytest.mark.parametrize('input, output, hidden_size', [
    (torch.randn((1, 64 * 64)), torch.randn((64 * 64)), 200),
    (torch.randn((1, 32 * 32)), torch.randn((32 * 32)), 300),
    (torch.randn((1, 128 * 128)), torch.randn((128 * 128)), 150),
    (torch.neg(torch.randn((1, 128 * 128))),
     torch.neg(torch.randn((128 * 128))), 150),  # negative tensors
    (torch.randn((1, 128 * 128)), torch.randn((32 * 32)), 100),
    # note that hidden size is >= 1
    (torch.empty((1, 999 * 999)), torch.empty((0, 0)), 1),
    (torch.Tensor([[]]), torch.Tensor([]), 1),  # empty input
    (torch.zeros(2, 3), torch.zeros(5), 1),
])
def test_lstm_io(input, output, hidden_size):
    """ Test the input and output shape of fully-connected LSTM
    >>> import torch
    >>> import tools.fc_lstm as fclstm
    >>> model = fclstm.LSTM(64*64,150,64*64)
    >>> X = torch.randn((1,64*64))
    >>> model(X).shape
    torch.Size([4096])
    """

    model = fclstm.LSTM(input.size(1), hidden_size, output.size(0))
    y_pred = model(input)
    assert y_pred.size(0) == output.size(0)


@pytest.mark.parametrize('model, optimizer, criterion, dataloader', [
    (dummy_model, dummy_optimizer, nn.MSELoss(), dummy_loader),
])
def test_train_lstm_mse(model, optimizer, criterion, dataloader):
    """ Test the training method of LSTM with MSE loss function"""

    loss = fclstm.train_lstm_mse(model, optimizer, criterion, dataloader)
    assert loss != 0


@pytest.mark.parametrize('model,criterion, dataloader', [
    (dummy_model, nn.MSELoss(), dummy_loader),
])
def test_validate_mse(model, criterion, dataloader):
    """ Test the validate method of LSTM with MSE loss function"""
    loss = fclstm.validate_mse(model, criterion, dataloader)
    assert loss != 0


@pytest.mark.parametrize('model, optimizer, criterion, dataloader', [
    (dummy_model, dummy_optimizer, ssim, dummy_loader),
])
def test_train_lstm_SSIM(model, optimizer, criterion, dataloader):
    """ Test the training method of LSTM with SSIM loss function"""
    result = fclstm.train_lstm_SSIM(model, optimizer, criterion, dataloader)
    assert result != 0


@pytest.mark.parametrize('model, criterion, dataloader', [
    (dummy_model, ssim, dummy_loader),
])
def test_validate_SSIM(model, criterion, dataloader):
    """ Test the validation method of LSTM with SSIM loss function"""
    result = fclstm.validate_SSIM(model, criterion, dataloader)
    assert result != 0


@pytest.mark.parametrize('model, dataloader', [
    (dummy_model, dummy_loader),
])
def test_evaluate(model, dataloader):
    """ Test the validation method of LSTM with SSIM loss function"""
    y_1, y_2 = fclstm.evaluate(model, dataloader)
    assert y_1.size != 0
    assert y_2.size != 0


@pytest.mark.parametrize(
    'seed,epochs,batch_size,model,criterion,optimizer,train_loader,val_loader,test_loader',
    [(42, 1, 1, dummy_model, nn.MSELoss(),
      dummy_optimizer, dummy_loader, dummy_loader, dummy_loader),
     (42, 10, 1, dummy_model, nn.MSELoss(),
      dummy_optimizer, dummy_loader, dummy_loader, dummy_loader), ])
def test_train_mse(seed, epochs, batch_size, model, criterion, optimizer,
                   train_loader, val_loader, test_loader):
    """ Test the training method of LSTM with SSIM loss function"""
    assert fclstm.train_mse(
        seed,
        epochs,
        batch_size,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        test_loader)


@pytest.mark.parametrize(
    'seed,epochs,batch_size,model,criterion,optimizer,train_loader,val_loader,test_loader',
    [
        (42,
         1,
         1,
         dummy_model,
         ssim,
         dummy_optimizer,
         dummy_loader,
         dummy_loader,
         dummy_loader),
        (42,
         10,
         1,
         dummy_model,
         ssim,
         dummy_optimizer,
         dummy_loader,
         dummy_loader,
         dummy_loader),
    ])
def test_train_ssim(seed, epochs, batch_size, model, criterion, optimizer,
                    train_loader, val_loader, test_loader):
    """ Test the training method of LSTM with SSIM loss function"""
    assert fclstm.train_ssim(
        seed,
        epochs,
        batch_size,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        test_loader)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
