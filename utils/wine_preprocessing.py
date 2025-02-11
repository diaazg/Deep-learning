# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# for number-crunching
import numpy as np
import scipy.stats as stats

# for dataset management
import pandas as pd

def load_wine_data():
    wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    return wine_data

def z_score_standardization(wine_df):
    wine_data = wine_df.copy()
    # z-score all columns except for quality
    cols2zscore = wine_data.columns.drop('quality')
    wine_data.loc[:, cols2zscore] = wine_data[cols2zscore].apply(stats.zscore)
    return wine_data

def binarize_wine_quality(wine_df):
    wine_data = wine_df.copy()
    # create new column for binarized quality
    wine_data.loc[:, 'boolQuality'] = 0
    wine_data.loc[wine_data['quality'] > 5, 'boolQuality'] = 1
    return wine_data

def preprocess_wine_data(wine_df):
    wine_data = wine_df.copy()
    # create a binary target variable
    wine_data = wine_data[wine_data['total sulfur dioxide'] < 200]
    wine_data = z_score_standardization(wine_data)
    wine_data = binarize_wine_quality(wine_data)
    return wine_data