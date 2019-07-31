import numpy as np
import fancyimpute
import pandas as pd
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer
import joblib


def create_typical(perc=10, c=0.6, extras=0):
    """
    Create a dataset with predicted values
    """
    full_data = np.random.multivariate_normal(
        [0 for i in range(n)], cov, size=size)
    df = pd.DataFrame(full_data)
    y = df[0] + df[1] + df[2] + np.random.standard_normal(size)
    return df, y


def create_missing(perc=10, c=0.6, extras=0):
    """
    Create a dataset with missing values and predicted values
    """
    n = 3 + extras
    cov = c + np.identity(n) * (1 - c)
    size = 100
    full_data = np.random.multivariate_normal(
        [0 for i in range(n)], cov, size=size)
    df = pd.DataFrame(full_data)
    y = df[0] + df[1] + df[2] + np.random.standard_normal(size)
    df_i = pd.DataFrame()
    for i in range(n):
        ind = np.zeros(size)
        missing = np.random.choice(100, size=perc)
        ind[missing] = 1
        df_i[i] = ind
        df[df_i == 1] = np.NAN
    return df, y


def test_drug(c=0.6, extras=0):
    """
    test dataset
    """
    n = 3 + extras
    cov = c + np.identity(n) * (1 - c)
    size = 20
    full_data = np.random.multivariate_normal(
        [0 for i in range(n)], cov, size=size)
    df = pd.DataFrame(full_data)
    y = df[0] + df[1] + df[2] + np.random.standard_normal(size)
    return df, y


def multi_imp_sum(perc, c, m=10):
    """multiple imputation for log prob"""
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    drug_dists = []
    prob_sum = 0  # guassian probs sum
    total_pred = np.zeros(20)
    total_var = np.zeros(20)
    for k in range(m):
        clf = BayesianRidge()
        design = fancyimpute.IterativeImputer(n_iter=10, sample_posterior=True, random_state=int(
            k * 243624) % 2**(32 - 1)).fit_transform(df)
        clf.fit(design, y)
        drug_preds, std = clf.predict(drug_vals, return_std=True)
        total_pred += drug_preds
        total_var += std**2
    prob_sum = (1 / m) * scipy.stats.norm(total_pred / m,
                                          (total_var / m)**0.5).pdf(drug_true).prod()
    return -np.log(prob_sum)
