import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def get_train(label=0, v=0,scale=False, *args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train", scale, v)

def get_test(label=0,v=0,scale=False, *args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test", scale,v)

def get_valid(label=0, scale=False, *args):
    """Get validation dataset for KDD 10 percent"""
    return None

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 121)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def get_anomalous_proportion():
    return 0.2
def _get_dataset(scale,v):
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
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train, y_train = _to_xy(df_train, target='label')
    x_test, y_test = _to_xy(df_test, target='label')

    y_train = y_train.flatten().astype(int)
    y_test = y_test.flatten().astype(int)
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]
    if scale:
        print("Scaling KDD dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset
def _get_dataset(scale,v):
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
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels
    if v == 0:
        df_train = df.sample(frac=0.5, random_state=42)
        df_test = df.loc[~df.index.isin(df_train.index)]

        x_train, y_train = _to_xy(df_train, target='label')
        x_test, y_test = _to_xy(df_test, target='label')

        y_train = y_train.flatten().astype(int)
        y_test = y_test.flatten().astype(int)
        x_train = x_train[y_train != 1]
        y_train = y_train[y_train != 1]
    else:
        dataset = df.sample(frac=1, random_state=42)
        features, labels = _to_xy(dataset, target='label')

        size_alldata = np.size(features,axis=0)

        size_traindata = size_alldata//2

        normal_x_data = features[labels == 0]
        normal_y_data = labels[labels == 0]
        size_nomaldata = normal_x_data.shape[0]

        anormal_x_data = features[labels == 1]
        anormal_y_data = labels[labels == 1]
        size_anormaldata = anormal_x_data.shape[0]
        size_tadata = int(v * size_anormaldata)
        size_tndata = size_traindata + size_tadata

        randNdata = np.arange(size_nomaldata)
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
    if scale:
        print("Scaling KDD dataset")
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


def _get_adapted_dataset(split, scale,v):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale,v)
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split == 'test':
        dataset[key_img], dataset[key_lbl] = _adapt_ratio(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])


def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().flatten().astype(int)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def _adapt_ratio(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    normal_ratio_test = int(len(np.where(y == 0)[0]) / len(y) * 100)


    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_x = inliersx.shape[0]
    out_size_x = int(size_x*rho/(1-rho))

    out_sample_x = outliersx[:out_size_x]
    out_sample_y = outliersy[:out_size_x]

    x_adapted = np.concatenate((inliersx,out_sample_x), axis=0)
    y_adapted = np.concatenate((inliersy,out_sample_y), axis=0)

    normal_ratio_test1 = int(len(np.where(y_adapted == 0)[0]) / len(y_adapted) * 100)


    size_x = x_adapted.shape[0]
    inds = rng.permutation(size_x)
    x_adapted, y_adapted = x_adapted[inds], y_adapted[inds]
    normal_ratio_test2 = int(len(np.where(y_adapted == 0)[0]) / len(y_adapted) * 100)

    return x_adapted, y_adapted




