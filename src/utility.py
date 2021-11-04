import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def read_monk(dataset_id):
    col_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Identifier']
    train_ds = pd.read_csv(f"../datasets/Monks/monks-{dataset_id}.train", sep=' ', names=col_names)
    train_ds.set_index('Identifier', inplace=True)

    train_ds = train_ds.sample(frac=1)
    labels = train_ds.pop('a0')

    train_ds = OneHotEncoder().fit_transform(train_ds).toarray().astype(np.float32)

    labels = labels.to_numpy()[:, np.newaxis]

    return train_ds, labels

def read_cup():
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 't1', 't2']
    train_ds = pd.read_csv(f"../datasets/Cup/ML-CUP21-TR.csv", sep=',', names=col_names)
    train_ds.set_index('Id', inplace=True)

    train_ds = train_ds.sample(frac=1)
    train = train_ds[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']].to_numpy()
    labels = train_ds[['t1']].to_numpy()

    train = StandardScaler().fit_transform(train)
    
    return train, labels