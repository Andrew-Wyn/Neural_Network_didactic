import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

def read_monk(dataset_id):
    col_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Identifier']
    train_ds = pd.read_csv(f"../datasets/Monks/monks-{dataset_id}.train", sep=' ', names=col_names)
    train_ds.set_index('Identifier', inplace=True)

    train_ds = train_ds.sample(frac=1)
    labels = train_ds.pop('a0')

    train_ds = OneHotEncoder().fit_transform(train_ds).toarray().astype(np.float32)

    labels = labels.to_numpy()[:, np.newaxis]

    return train_ds, labels