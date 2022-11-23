import torch
import torch.optim
import tools.fc_lstm_pca as fclstm_pca
import tools.fc_lstm as fclstm
from tools.dataprocessing import create_inout_seq_pca
from torch.utils.data import TensorDataset, DataLoader
import pytest
from sklearn.decomposition import PCA

# n_components = min(n_components,n_features) - 1
pca_trained = PCA(n_components=93)

train = torch.stack([torch.randn((100 * 100))] * 100)
test = torch.stack([torch.randn((100 * 100))] * 100)

train = pca_trained.fit_transform(train)
test = pca_trained.transform(test)

train_data = torch.from_numpy(train).float()
test_data = torch.from_numpy(test).float()

test_ds, test_label = create_inout_seq_pca(test_data, 1)  # (99,1,93)

# dummy_data = TensorDataset(train_data, test_data)
dummy_data = TensorDataset(test_ds, test_label)
dummy_loader = DataLoader(dummy_data, batch_size=1,
                          shuffle=True, num_workers=0)
dummy_model = fclstm.LSTM(93, 100, 93)


@pytest.mark.parametrize('model,dataloader,pca_trained', [
    (dummy_model, dummy_loader, pca_trained),
])
def test_evaluate_pca(model, dataloader, pca_trained):
    """ Test validation method of LSTM with PCA
    """

    y_1, y_2 = fclstm_pca.evaluate_pca(model, dataloader, pca_trained)
    assert y_1.size != 0
    assert y_2.size != 0
    assert len(y_2) == len(dataloader)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
