import pkg_resources

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def get_preprocess_monk(stream):
    col_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Identifier']
    ds = pd.read_csv(stream, sep=' ', names=col_names)
    ds.set_index('Identifier', inplace=True)

    ds = ds.sample(frac=1)
    labels = ds.pop('a0')

    #ds = OneHotEncoder().fit_transform(ds).toarray().astype(np.float32)
    ds = ds.to_numpy()
    
    labels = labels.to_numpy()[:, np.newaxis]

    return ds, labels


def read_monk(dataset_id):
    stream_train = pkg_resources.resource_stream(__name__, f"data/Monks/monks-{dataset_id}.train")
    stream_test = pkg_resources.resource_stream(__name__, f"data/Monks/monks-{dataset_id}.test")
    
    train_ds, train_labels = get_preprocess_monk(stream_train)
    test_ds, test_labels = get_preprocess_monk(stream_test)

    return train_ds, test_ds, train_labels, test_labels
    

def read_cup():
    stream = pkg_resources.resource_stream(__name__, f"data/Cup/ML-CUP21-TR.csv")

    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 't1', 't2']
    train_ds = pd.read_csv(stream, sep=',', names=col_names)
    train_ds.set_index('Id', inplace=True)

    train_ds = train_ds.sample(frac=1)
    train = train_ds[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']].to_numpy()
    labels = train_ds[['t1', 't2']].to_numpy()

    train = StandardScaler().fit_transform(train)
    
    return train, labels