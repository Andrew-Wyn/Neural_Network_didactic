import pkg_resources

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def get_preprocess_monk(stream, preprocesser=None):
    """
    Preprocess monk dataset by OneHotEncoder of sklearn
    
    Args:
        stream: (IO[bytes]) stream of the file in which the dataset resides
        preprocesser: (OneHotEncoder) object of the type of preprocesser of sklearn if None will be fitted on this dataset
    Returns:
        ds: (np.ndarray) dataset numpy
        target: (np.ndarray) target numpy
        preprocesser: (OneHotEncoder) 
    """

    col_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Identifier']
    ds = pd.read_csv(stream, sep=' ', names=col_names)
    ds.set_index('Identifier', inplace=True)

    ds = ds.sample(frac=1)
    target = ds.pop('a0')

    if not preprocesser:
        preprocesser = OneHotEncoder().fit(ds)

    ds = preprocesser.transform(ds).toarray().astype(np.float32)

    target = target.to_numpy()[:, np.newaxis]

    return ds, target, preprocesser


def get_preprocess_cup(stream, preprocesser=None):
    """
    Preprocess cup dataset by StandardScaler of sklearn
    
    Args:
        stream: (IO[bytes]) stream of the file in which the dataset resides
        preprocesser: (StandardScaler) object of the type of preprocesser of sklearn if None will be fitted on this dataset
    Returns:
        ds: (np.ndarray) dataset numpy
        target: (np.ndarray) target numpy
        preprocesser: (OneHotEncoder) 
    """

    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 't1', 't2']
    ds = pd.read_csv(stream, sep=',', names=col_names)
    ds.set_index('Id', inplace=True)

    ds = ds.sample(frac=1)
    target = ds[['t1', 't2']].copy().to_numpy()

    ds.drop(['t1', 't2'], axis=1, inplace=True)

    if not preprocesser:
        preprocesser = StandardScaler().fit(ds)

    ds = preprocesser.transform(ds)
    
    return ds, target, preprocesser


def read_monk(dataset_id):
    """
    It reads the monk dataset and return training and test datasets in numpy format
    
    Args:
        dataset_id: (int) refers to the id of the datasets: monk-1, monk-2, monk-3
    Returns:
        train_ds: (np.ndarray) training dataset
        test_ds: (np.ndarray) test dataset
        train_target: (np.ndarray) training target
        test_target: (np.ndarray) test target
        preprocesser: (OneHotEncoder)
    """

    stream_train = pkg_resources.resource_stream(__name__, f"data/Monks/monks-{dataset_id}.train")
    stream_test = pkg_resources.resource_stream(__name__, f"data/Monks/monks-{dataset_id}.test")
    
    train_ds, train_target, preprocesser = get_preprocess_monk(stream_train)
    test_ds, test_target, _ = get_preprocess_monk(stream_test, preprocesser)

    return train_ds, test_ds, train_target, test_target, preprocesser
    

def read_cup():
    """
    It reads the cup dataset and return training and test datasets in numpy format

    Returns:
        train_ds: (np.ndarray) training dataset
        test_ds: (np.ndarray) test dataset
        train_target: (np.ndarray) training target
        test_target: (np.ndarray) test target
        preprocesser: (StandardScaler)
    """

    stream_train = pkg_resources.resource_stream(__name__, f"data/Cup/cup_train.csv")
    stream_test = pkg_resources.resource_stream(__name__, f"data/Cup/cup_test.csv")

    train_ds, train_target, preprocesser = get_preprocess_cup(stream_train)
    test_ds, test_target, _ = get_preprocess_cup(stream_test, preprocesser)

    return train_ds, test_ds, train_target, test_target, preprocesser


def read_cup_blind_test(preprocesser):
    """
    It reads the cup blind test dataset

    Args:
        preprocesser: (StandardScaler) same standard scaler over the training dataset
    Returns:
        ds: (np.ndarray) dataset in a numpy format
    """

    stream = pkg_resources.resource_stream(__name__, f"data/Cup/cup_blind_test.csv")

    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']
    ds = pd.read_csv(stream, sep=',', names=col_names)
    ds.set_index('Id', inplace=True)

    ds = preprocesser.transform(ds)
    
    return ds