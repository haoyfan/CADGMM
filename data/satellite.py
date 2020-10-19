import logging
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def get_train(label=0, scale=False, v=0, *args):
    """Get training dataset for Thyroid dataset"""
    return _get_adapted_dataset("train", scale, v=v)


def get_test(label=0, scale=False, v=0, *args):
    """Get testing dataset for Thyroid dataset"""
    return _get_adapted_dataset("test", scale, v)


def get_valid(label=0, scale=False, *args):
    """Get validation dataset for Thyroid dataset"""
    return None


def get_shape_input():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 36)


def get_shape_input_flatten():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 36)


def get_shape_label():
    """Get shape of the labels in Thyroid dataset"""
    return (None,)


def get_anomalous_proportion():
    return 0.32


# def _get_dataset(scale):
#     """ Gets the basic dataset
#     Returns :
#             dataset (dict): containing the data
#                 dataset['x_train'] (np.array): training images shape
#                 (?, 120)
#                 dataset['y_train'] (np.array): training labels shape
#                 (?,)
#                 dataset['x_test'] (np.array): testing images shape
#                 (?, 120)
#                 dataset['y_test'] (np.array): testing labels shape
#                 (?,)
#     """
#     data = scipy.io.loadmat("data/satellite.mat")
#
#     full_x_data = data["X"]
#     full_y_data = data['y']
#     x_train, x_test, \
#     y_train, y_test = train_test_split(full_x_data,
#                                        full_y_data,
#                                        test_size=0.5,
#                                        random_state=42)
#
#     y_train = y_train.flatten().astype(int)
#     y_test = y_test.flatten().astype(int)
#
#     if scale:
#         print("Scaling dataset")
#         scaler = MinMaxScaler()
#         scaler.fit(x_train)
#         x_train = scaler.transform(x_train)
#         x_test = scaler.transform(x_test)
#
#     dataset = {}
#     dataset['x_train'] = x_train.astype(np.float32)
#     dataset['y_train'] = y_train.astype(np.float32)
#     dataset['x_test'] = x_test.astype(np.float32)
#     dataset['y_test'] = y_test.astype(np.float32)
#
#     return dataset
#
#     return dataset


def _get_dataset(scale, v):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    data = scipy.io.loadmat("data/satellite.mat")

    full_x_data = data["X"]
    full_y_data = data['y']

    y_data = full_y_data.flatten().astype(int)

    normal_x_data = full_x_data[y_data != 1]
    normal_y_data = full_y_data[y_data != 1]
    anormal_x_data = full_x_data[y_data == 1]
    anormal_y_data = full_y_data[y_data == 1]

    size_alldata = np.size(full_x_data, axis=0)
    size_normaldata = np.size(normal_x_data, axis=0)
    size_anormaldata = np.size(anormal_x_data, axis=0)
    size_traindata = size_alldata // 2
    size_tadata = v * size_anormaldata  # size of anomal data in train data
    size_tndata = size_traindata - size_tadata  # size of nomal data in train data
    randNdata = np.arange(size_normaldata)
    randAdata = np.arange(size_anormaldata)
    np.random.shuffle(randNdata)
    np.random.shuffle(randAdata)
    x_tndata = normal_x_data[randNdata[:size_tndata]]
    y_tndata = normal_y_data[randNdata[:size_tndata]]
    x_tadata = anormal_x_data[randAdata[:size_tadata]]
    y_tadata = anormal_y_data[randAdata[:size_tadata]]

    x_tendata = normal_x_data[randNdata[size_tndata:]]
    y_tendata = normal_y_data[randNdata[size_tndata:]]
    x_teadata = anormal_x_data[randAdata[size_tadata:]]
    y_teadata = anormal_y_data[randAdata[size_tadata:]]
    x_train = np.concatenate((x_tndata, x_tadata), axis=0)
    y_train = np.concatenate((y_tndata, y_tadata), axis=0)
    N_train = x_train.shape[0]
    randIdt = np.arange(N_train)
    np.random.shuffle(randIdt)
    x_train = x_train[randIdt[:]]
    y_train = y_train[randIdt[:]]

    x_test = np.concatenate((x_tendata, x_teadata), axis=0)
    y_test = np.concatenate((y_tendata, y_teadata), axis=0)
    N_test = x_test.shape[0]
    randIdt = np.arange(N_test)
    np.random.shuffle(randIdt)
    x_test = x_test[randIdt[:]]
    y_test = y_test[randIdt[:]]
    y_test = np.squeeze(y_test)

    if scale:
        print("Scaling dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    normal_ratio_train = int(len(np.where(y_train == 0)[0]) / len(y_train) * 100)
    normal_ratio_test = int(len(np.where(y_test == 0)[0]) / len(y_test) * 100)



    return dataset


def _get_adapted_dataset(split, scale, v):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale, v)
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    print("Size of split", split, ":", dataset[key_lbl].shape[0])

    return (dataset[key_img], dataset[key_lbl])


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

