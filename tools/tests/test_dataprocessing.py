# from pkg_resources import ResolutionError
import tools.dataprocessing as dataprocessing
# import numpy as np
import pytest
import pandas as pd
# from torch.utils.data import TensorDataset, DataLoader


@pytest.mark.parametrize('train_source, train_labels, download_from, res', [
    ("some_invalid_paths", "some_invalid_paths", "./",
     "Can't find the dataset, please set the directory again!"),
])
def test_Load_data_to_train(train_source, train_labels, download_from, res):
    """ Test the Load to train function """
    result = dataprocessing.Load_data_to_train(
        train_source, train_labels, download_from)
    # print(result)
    assert result == res


@pytest.mark.parametrize('test_source, download_from, res', [
    ("some_invalid_paths", "./",
     "Can't find the dataset, please set the directory again!"),
])
def test_Load_data_to_test(test_source, download_from, res):
    """ Test the load to test function """
    result = dataprocessing.Load_data_to_test(test_source, download_from)
    # print(result)
    assert result == res


df1 = pd.DataFrame({"x": [5, 2], "y": [4, 7], "z": [9, 3]})
df2 = pd.DataFrame({"x": [1, 3], "y": [1, 9], "z": [29, 30]})
answer1 = "Image ID doesn't exist in dataframe"

df3 = pd.DataFrame({"Image ID": ["5", "2"], "y": [4, 7], "z": [9, 3]})
df4 = pd.DataFrame({"Image ID": ["1", "3"], "y": [1, 9], "z": [29, 30]})
answer2 = "Storm ID doesn't exist in dataframe"


@pytest.mark.parametrize('train_df, test_df, res', [
    (df1, df2, answer1),
    (df3, df4, answer2),
])
def test_manipdata(train_df, test_df, res):
    """ Test the manipdata function """
    result = dataprocessing.manipdata(train_df, test_df)
    assert res == result


@pytest.mark.parametrize('full_df, id, res', [
    (df1, "abc", answer2),
    (df2, "abc", answer2),
])
def test_data_storm_id(full_df, id, res):
    """ Test the manipdata function """
    result = dataprocessing.data_storm_id(full_df, id)
    assert res == result

# x = DataLoader((1,"2"))
# @pytest.mark.parametrize('loader, dimension, res', [
#     (x, 2, 'The data is not in the right format'),
# ])

# def test_Loader_to_1D_array(loader, dimension, res):
#     """ Test the manipdata function """
#     result = dataprocessing.Loader_to_1D_array(loader, dimension)
#     assert res == result


x = {1: 2}


@pytest.mark.parametrize('input_data, tw, res', [
    (x, 2, "Please insert a list"),
])
def test_create_inout_seq(input_data, tw, res):
    """ Test the manipdata function """
    result = dataprocessing.create_inout_seq(input_data, tw)
    assert res == result


# @pytest.mark.parametrize("inputs, res", [
#     (1, "Wrong input, please input Pandas as x_full_id and y_full_id"),
# ])

# def test_SplitData(inputs, res):
#     """ Test the manipdata function """
#     result = dataprocessing.SplitData(inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs)
#     print("hi", result)
#     assert res == result


# @pytest.mark.parametrize("inputs, res", [
#     ([1,2,3], "Input training data is not in the right format"),
# ])

# def test_DatasetSTORM(inputs, res):
#     """ Test the manipdata function """
#     result = dataprocessing.DatasetSTORM(inputs)
#     print("hi", result)
#     assert res == result


if __name__ == "__main__":
    import doctest
    doctest.testmod()
