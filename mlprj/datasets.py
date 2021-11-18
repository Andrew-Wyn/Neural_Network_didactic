import pkg_resources

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def read_monk(dataset_id):
    stream = pkg_resources.resource_stream(__name__, f"data/Monks/monks-{dataset_id}.train")
    
    col_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Identifier']
    train_ds = pd.read_csv(stream, sep=' ', names=col_names)
    train_ds.set_index('Identifier', inplace=True)

    train_ds = train_ds.sample(frac=1)
    labels = train_ds.pop('a0')

    train_ds = OneHotEncoder().fit_transform(train_ds).toarray().astype(np.float32)

    labels = labels.to_numpy()[:, np.newaxis]

    return train_ds, labels

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