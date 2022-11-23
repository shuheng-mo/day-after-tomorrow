from glob import glob
from pathlib import Path
import json
import pandas as pd
import numpy as np
from torchvision import transforms
import torch
from PIL import Image as pil_image


__all__ = [
    'Load_data_to_train',
    "Load_data_to_test",
    "manipdata",
    "data_storm_id",
    "Loader_to_1D_array",
    "SplitData",
    "DatasetSTORM",
    "create_inout_seq"]


def Load_data_to_train(train_source, train_labels, download_from):
    """
    The function takes in the location of the download directory where we installed
    all the data and the train_source directs us to the location of the data and train_labels
    directs us to the wind_speed label. The function loads all of this data and put them
    together in a dataframe and returns it.

    Parameters
    ----------
    train_source : string
    train_labels : string
    download_from : string

    Returns
    ----------
    pandas dataframe that merges all these files
    together

    Examples
    --------
    >>> train_source = 'some_invalid_paths'
    >>> train_labels = "some_invalid_paths"
    >>> download_from = './'
    >>> failed_load = Load_data_to_train(train_source, train_labels, download_from)
    >>> failed_load
    "Can't find the dataset, please set the directory again!"

    Notes
    -----
    See
    https://nbviewer.org/github/radiantearth/mlhub-tutorials/blob/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge/nasa-tropical-storm-wind-speed-challenge-getting-started.ipynb
    for further details.
    """

    train_data = []
    # train_source = 'nasa_tropical_storm_competition_train_source'
    # train_labels = 'nasa_tropical_storm_competition_train_labels'
    download_dir = Path(download_from)
    jpg_names = glob(str(download_dir / train_source / '**' / '*.jpg'))
    if jpg_names == []:
        return "Can't find the dataset, please set the directory again!"
    for jpg_path in jpg_names:
        jpg_path = Path(jpg_path)
        # Get the IDs and file paths
        features_path = jpg_path.parent / 'features.json'
        image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
        storm_id = image_id.split('_')[0]
        labels_path = str(
            jpg_path.parent /
            'labels.json').replace(
            train_source,
            train_labels)
        # Load the features data
        with open(features_path) as src:
            features_data = json.load(src)
        # Load the labels data
        with open(labels_path) as src:
            labels_data = json.load(src)

        train_data.append([
            image_id,
            storm_id,
            int(features_data['relative_time']),
            int(features_data['ocean']),
            int(labels_data['wind_speed'])
        ])
    train_df = pd.DataFrame(
        np.array(train_data),
        columns=[
            'Image ID',
            'Storm ID',
            'Relative Time',
            'Ocean',
            'Wind Speed']).sort_values(
        by=['Image ID']).reset_index(
                drop=True)
    return train_df


def Load_data_to_test(test_source, download_from):
    """
    The function takes in the location of the download directory where we installed
    all the data and the test_source directs us to the location of the data. The function
    loads all of this data and put them together in a dataframe and returns it

    Parameters
    ----------
    test_source : string
    download_from : string

    Returns
    ----------
    pandas dataframe that merges all these files
    together

    Examples
    --------
    >>> test_source = 'some_invalid_paths'
    >>> download_from = './'
    >>> failed_load = Load_data_to_test(test_source, download_from)
    >>> failed_load
    "Can't find the dataset, please set the directory again!"

    Notes
    -----
    See
    https://nbviewer.org/github/radiantearth/mlhub-tutorials/blob/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge/nasa-tropical-storm-wind-speed-challenge-getting-started.ipynb
    for further details.
    """
    test_data = []
    download_dir = Path(download_from)
    jpg_names = glob(str(download_dir / test_source / '**' / '*.jpg'))
    if jpg_names == []:
        return "Can't find the dataset, please set the directory again!"
    for jpg_path in jpg_names:
        jpg_path = Path(jpg_path)

        # Get the IDs and file paths
        features_path = jpg_path.parent / 'features.json'
        image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
        storm_id = image_id.split('_')[0]

        # Load the features data
        with open(features_path) as src:
            features_data = json.load(src)

        test_data.append([
            image_id,
            storm_id,
            int(features_data['relative_time']),
            int(features_data['ocean']),
        ])

    test_df = pd.DataFrame(
        np.array(test_data),
        columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean']
    ).sort_values(by=['Image ID']).reset_index(drop=True)

    # test_df.head()
    test_df.sort_values(by='Relative Time')
    return test_df


def manipdata(train_df, test_df):
    """
    The function takes in training and testing data as a dataframe. Then,
    combines training and testing data, add filename as a column with the suitable path depending on the image id
    and adds the number of images per storm as the final column. The whole dataframe is returned

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame


    Returns
    ----------
    pandas dataframe that merges all these files
    together

    Examples
    --------
    >>> df1 = pd.DataFrame({"A": [5, 2], "B": [4, 7], "Z": [9, 3]})
    >>> df2 = pd.DataFrame({"A": [1, 3], "B": [1, 9], "Z": [29, 30]})
    >>> failed_manip = manipdata(df1, df2)
    >>> failed_manip
    "Image ID doesn't exist in dataframe"

    >>> df3 = pd.DataFrame({"Image ID": ["5", "2"], "B": [4, 7], "C": [9, 3]})
    >>> df4 = pd.DataFrame({"Image ID": ["1", "3"], "B": [1, 9], "C": [29, 30]})
    >>> failed_manip2 = manipdata(df3, df4)
    >>> failed_manip2
    "Storm ID doesn't exist in dataframe"

    Notes
    -----
    See
    https://nbviewer.org/github/radiantearth/mlhub-tutorials/blob/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge/nasa-tropical-storm-wind-speed-challenge-getting-started.ipynb
    for further details.

    """
    # Combine dataset
    full_df = train_df.append(test_df)

    # Add file name
    if "Image ID" in full_df:
        full_df["file_name"] = (
            "./nasa_tropical_storm_competition_train_source/nasa_tropical_storm_competition_train_source_" +
            full_df['Image ID'] +
            '/image.jpg')
    else:
        return "Image ID doesn't exist in dataframe"

    # Add Images per storm
    if "Storm ID" in full_df:
        images_per_storm = full_df.groupby(
            "Storm ID").size().to_frame("images_per_storm")
        full_df = full_df.merge(images_per_storm, how="left", on="Storm ID")
    else:
        return "Storm ID doesn't exist in dataframe"
    return full_df


def manipdata_surprise(train_df):
    """
    The function takes in training and testing data as a dataframe. Then,
    combines training and testing data, add filename as a column with the suitable path depending on the image id
    and adds the number of images per storm as the final column. The whole dataframe is returned

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame


    Returns
    ----------
    pandas dataframe that merges all these files
    together

    Examples
    --------
    >>> df1 = pd.DataFrame({"A": [5, 2], "B": [4, 7], "Z": [9, 3]})
    >>> df2 = pd.DataFrame({"A": [1, 3], "B": [1, 9], "Z": [29, 30]})
    >>> failed_manip = manipdata(df1, df2)
    >>> failed_manip
    "Image ID doesn't exist in dataframe"

  >>> df3 = pd.DataFrame({"Image ID": ["5", "2"], "B": [4, 7], "C": [9, 3]})
  >>> df4 = pd.DataFrame({"Image ID": ["1", "3"], "B": [1, 9], "C": [29, 30]})
  >>> failed_manip2 = manipdata(df3, df4)
  >>> failed_manip2
  "Storm ID doesn't exist in dataframe"

  Notes
  -----
  See
  https://nbviewer.org/github/radiantearth/mlhub-tutorials/blob/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge/nasa-tropical-storm-wind-speed-challenge-benchmark.ipynb
  for further details.
  """

    # Add file name
    if "Image ID" in train_df:
        train_df["file_name"] = (
            "./nasa_tropical_storm_competition_surprise_storm_source/nasa_tropical_storm_competition_surprise_storm_source_" +
            train_df['Image ID'] +
            '/image.jpg')
    else:
        return "Image ID doesn't exist in dataframe"

    # Add Images per storm
    if "Storm ID" in train_df:
        images_per_storm = train_df.groupby(
            "Storm ID").size().to_frame("images_per_storm")
        train_df = train_df.merge(images_per_storm, how="left", on="Storm ID")
    else:
        return "Storm ID doesn't exist in dataframe"
    return train_df


def data_storm_id(full_df, id):
    """
    given a dataframe and a storm id as a string, return the dataframe with the corresponding ID

    Parameters
    ----------
    full_df : pd.DataFrame
    id : string


    Returns
    ----------
    return a pandas dataframe with corresponding ID

    Examples
    --------
    >>> df1 = pd.DataFrame({"A": [5, 2], "B": [4, 7], "Z": [9, 3]})
    >>> failed_datastormid = data_storm_id(df1, 2)
    >>> failed_datastormid
    "Storm ID doesn't exist in dataframe"

    """
    if "Storm ID" in full_df:
        return full_df[full_df["Storm ID"] == id]
    else:
        return "Storm ID doesn't exist in dataframe"


def Loader_to_1D_array(loader, dimension):
    """
    given a data loader (ie. train, test, validation) and the dimension of 1D array
    return 3 arrays containing the images, the labels(wind speed) and the image_id

    Parameters
    ----------
    loader : torch.utils.data.dataloader.DataLoader
    dimension : float/int

    Returns
    ----------
    return three lists
    """
    for data in loader:
        if "Image ID" not in data or "image" not in data or "label" not in data:
            return "The data is not in the right format"
    image_id = [data["Image ID"] for data in loader]
    images = [data["image"].reshape(1, dimension)
              for data in loader]  # Convert to 1D array
    labels = [data["label"] for data in loader]
    return image_id, images, labels


def SplitData(
        x_full_id,
        y_full_id,
        train_size,
        val_size,
        test_size,
        cleaning_class,
        old_mean,
        old_std):
    """
    Given the dataframe for training and labelling (wind speed), the sizes of training, testing, validation data
    and the class for cleaning the data with the original mean and standard deviation. The data is splitted into
    the full dataset, training dataset, validation and testing dataset as

    Parameters
    ----------
    x_full_id: pd.DataFrame
    y_full_id: pd.DataFrame
    train_size: float/int
    val_size: float/int
    test_size: float/int
    cleaning_class: float/int
    old_mean: float/int
    old_std: float/int

    Returns
    ----------
    return two torch tensors one which will be trained and the other for prediction
    """
    df_type_wrong1 = not isinstance(x_full_id, type(pd.DataFrame({})))
    df_type_wrong2 = not isinstance(y_full_id, type(pd.Series({})))
    df_type_wrong = df_type_wrong1 or df_type_wrong2
    if df_type_wrong:
        return "Wrong input, please input Pandas as x_full_id and y_full_id"
    full_dataset = cleaning_class(x_full_id, y_full_id, old_mean, old_std)
    val_dataset = cleaning_class(
        x_full_id[:val_size], y_full_id[:val_size], old_mean, old_std)
    train_dataset = cleaning_class(
        x_full_id[val_size: val_size + train_size],
        y_full_id[val_size: val_size + train_size],
        old_mean, old_std)
    test_dataset = cleaning_class(
        x_full_id
        [val_size + train_size - 1: val_size + train_size + test_size],
        y_full_id
        [val_size + train_size - 1: val_size + train_size + test_size],
        old_mean, old_std)
    print(
        "full data: ",
        len(full_dataset),
        "\nTrain_size: ",
        len(train_dataset),
        "\nval_size:",
        len(val_dataset),
        "\ntest_size: ",
        len(test_dataset))
    return full_dataset, train_dataset, val_dataset, test_dataset


def create_inout_seq(input_data, tw):
    """
    given a list of input_data and the future prediction number,
    we return the torch tensors of training data that takes tw datapoints and the corresponding
    labels which takes the next datapoint which will be used for prediction.

    Parameters
    ----------
    input_data : list
    tw : float/int

    Returns
    ----------
    return two torch tensors one which will be trained and the other for prediction

    Notes
    -----
    see
    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
    for further details
    """
    if not isinstance(input_data, type([])):
        return "Please insert a list"
    L = len(input_data)
    train_data = []
    label_data = []
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        train_data.append(torch.stack(train_seq))
        label_data.append(torch.stack(train_label))
    return torch.stack(train_data), torch.stack(label_data)


def create_inout_seq_pca(input_data, tw):
    """
    given a list of input_data and the future prediction number,
    we return the torch tensors of training data that takes tw datapoints and the corresponding
    labels which takes the next datapoint which will be used for prediction.

    Parameters
    ----------
    input_data : list
    tw : float/int

    Returns
    ----------
    return two torch tensors one which will be trained and the other for prediction

    Notes
    -----
    see
    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
    for further details
    """
    # if type(input_data) is not type([]):
    #   return "Please insert a list"
    L = len(input_data)
    train_data = []
    label_data = []
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        train_data.append(train_seq)
        label_data.append(train_label)
    return torch.stack(train_data), torch.stack(label_data)


class DatasetSTORM():
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.


    Parameters
    ----------
    x_train : pd.DataFrame
    y_train : pd.DataFrame
    mean_ : float/int
    std_ : float/int

    Notes
    -----
    see
    https://nbviewer.org/github/radiantearth/mlhub-tutorials/blob/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge/nasa-tropical-storm-wind-speed-challenge-benchmark.ipynb

    """

    def __init__(self, x_train, y_train=None, mean_=0, std_=1):
        self.data = x_train
        self.label = y_train
        self.mean_ = mean_
        self.std_ = std_
        self.transform = transforms.Compose(
            [
                transforms.Resize([366, 366]),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_, std=std_),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # if self.data is not type(pd.DataFrame):
        #   return "Input training data is not in the right format"
        image = pil_image.open(
            self.data.iloc[index]["file_name"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.iloc[index]["Image ID"]
        if self.label is not None:
            label = self.label.iloc[index]
            sample = {"Image ID": image_id, "image": image, "label": label}
        else:
            sample = {
                "Image ID": image_id,
                "image": image,
            }
        return sample


if __name__ == "__main__":
    import doctest
    doctest.testmod()
