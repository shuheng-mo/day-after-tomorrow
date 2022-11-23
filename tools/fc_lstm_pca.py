import numpy as np
import torch

__all__ = ["evaluate_pca"]

# specify the device
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


def evaluate_pca(model, data_loader, trained_pca):
    """
    Evaluate the model on test data_loader for after PCA. This
    function will return two numpy arrays one for prediction and
    the other for the actual solution

    Parameters
    ----------
    model: nn.Module
    data_loader: torch.utils.data.DataLoader
    trained_pca: sklearn.decomposition.PCA

    Returns
    ----------
    Resultant numpy arrays (numpy.ndarray)

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import tools.fc_lstm as fclstm
    >>> import tools.fc_lstm_pca as fclstm_pca
    >>> from tools.dataprocessing import create_inout_seq_pca
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> from sklearn.decomposition import PCA
    >>> pca_trained = PCA(n_components=93) # n_components = min(n_components,n_features) - 1
    >>> train = torch.stack([torch.randn((100*100))] * 100)
    >>> test = torch.stack([torch.randn((100*100))] * 100)
    >>> train = pca_trained.fit_transform(train)
    >>> test = pca_trained.transform(test)
    >>> train_data = torch.from_numpy(train).float()
    >>> test_data = torch.from_numpy(test).float()
    >>> test_ds,test_label = create_inout_seq_pca(test_data,1) # (99,1,93)
    >>> dummy_data = TensorDataset(test_ds, test_label)
    >>> dummy_loader = DataLoader(dummy_data, batch_size=1,shuffle=True, num_workers=0)
    >>> model = fclstm.LSTM(93, 100, 93)
    >>> y_1, y_2 = fclstm_pca.evaluate_pca(model, dummy_loader, pca_trained)
    >>> y_1.size != 0
    True
    >>> y_2.size != 0
    True

    Returns
    ----------
    Two numpy arrays one for output of model and the other for ground truth image
    """
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = trained_pca.inverse_transform(
                y_pred.cpu().detach().numpy())
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred)
    return np.concatenate(y_preds, 0), np.concatenate(ys, 0)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
