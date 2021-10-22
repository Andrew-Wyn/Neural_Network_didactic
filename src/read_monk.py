import pandas as pd

def read_monk_1():
    monk1_train = pd.read_csv('../datasets/Monks/monks-1.train', sep=' ')
    monk1_test = pd.read_csv('../datasets/Monks/monks-1.test', sep=' ')

    return monk1_train, monk1_test

def read_monk_2():
    monk2_train = pd.read_csv('../datasets/Monks/monks-2.train', sep=' ')
    monk2_test = pd.read_csv('../datasets/Monks/monks-1.test', sep=' ')

    return monk2_train, monk2_test

def read_monk_3():
    monk3_train = pd.read_csv('../datasets/Monks/monks-3.train', sep=' ')
    monk3_test = pd.read_csv('../datasets/Monks/monks-1.test', sep=' ')

    return monk3_train, monk3_test
