import time
import glm_testing
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import fancyimpute
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
import scipy
from multiprocessing import Pool
import os
import workers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def log_multi(perc, c, m=10, extras=0):
    """
    Logistic regression classifier with multiple imputation
    """
    clf = LogisticRegression(solver="liblinear")
    df, y = glm_testing.create_missing(perc=perc, c=c, extras=extras)
    drug_vals, drug_true = glm_testing.test_drug(c=c, extras=extras)
    y = y.apply(lambda x: 1 if x > 1 else 0)
    drug_true = drug_true.apply(lambda x: 1 if x > 1 else 0)
    drug_dists = []
    prob_sum = 0  # guassian probs sum
    for k in range(m):
        design = fancyimpute.IterativeImputer(n_iter=10, sample_posterior=True, random_state=int(
            k * 243624) % 2**(32 - 1)).fit_transform(df)
        clf.fit(design, y)
        prob_sum += (1 / m) * (clf.predict_proba(drug_vals)[:, 1][drug_true == 1].prod(
        ) * clf.predict_proba(drug_vals)[:, 0][drug_true == 0].prod())
    return np.log(prob_sum)


def log_mean(perc, c, extras=0):
    """
    Logistic regression with mean imputation
    """
    clf = LogisticRegression(solver="liblinear")
    df, y = glm_testing.create_missing(perc=perc, c=c, extras=extras)
    y = y.apply(lambda x: 1 if x > 1 else 0)
    drug_vals, drug_true = glm_testing.test_drug(c=c, extras=extras)
    drug_true = drug_true.apply(lambda x: 1 if x > 1 else 0)
    design = fancyimpute.SimpleFill(fill_method='mean').fit_transform(df)
    clf.fit(design, y)
    return (np.log(clf.predict_proba(drug_vals)[:, 1][drug_true == 1].prod() *
                   clf.predict_proba(drug_vals)[:, 0][drug_true == 0].prod()))


def dec_multi(perc, c, m=10):
    """
    Decision tree classifier with multiple imputation
    """
    clf = DecisionTreeClassifier()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    y = y.apply(lambda x: 1 if x > 1 else 0)
    drug_true = drug_true.apply(lambda x: 1 if x > 1 else 0)
    drug_dists = []
    prob_sum = 0
    for k in range(m):
        design = fancyimpute.IterativeImputer(n_iter=10, sample_posterior=True, random_state=int(
            k * 243624) % 2**(32 - 1)).fit_transform(df)
        clf.fit(design, y)
        prob_sum += (1 / m) * (clf.predict_proba(drug_vals)[:, 1][drug_true == 1].prod(
        ) * clf.predict_proba(drug_vals)[:, 0][drug_true == 0].prod())
    return np.log(prob_sum)


def dec_mean(perc, c):
    """
    Decision tree classifier with mean imputation
    """
    clf = DecisionTreeClassifier()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    y = y.apply(lambda x: 1 if x > 1 else 0)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    drug_true = drug_true.apply(lambda x: 1 if x > 1 else 0)
    design = fancyimpute.SimpleFill(fill_method='mean').fit_transform(df)
    clf.fit(design, y)
    return (np.log(clf.predict_proba(drug_vals)[:, 1][drug_true == 1].prod() *
                   clf.predict_proba(drug_vals)[:, 0][drug_true == 0].prod()))


def drop_vals(perc, c):
    """ drop vals giving log prob"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = df.drop_na()
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return -scipy.stats.norm(drug_preds, std * 1).logpdf(drug_true).sum()


def drop_vals_conf(perc, c):
    """drop vals that give conf"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = df.drop_na()
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return -scipy.stats.norm(drug_preds, std * 1).logpdf(drug_true).sum()


def mean_fill(perc, c):
    """mean fill that provies the log probability"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = fancyimpute.SimpleFill(fill_method='mean').fit_transform(df)
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return -scipy.stats.norm(drug_preds, std * 1).logpdf(drug_true).sum()


def mean_fill_conf(perc, c):
    """mean fill that provides the conf intervals"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = fancyimpute.SimpleFill(fill_method='mean').fit_transform(df)
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return sum(
        scipy.stats.norm(
            drug_preds,
            std *
            1).pdf(drug_true) < 0.05)  # 95 percent confidence


def knn_fill(perc, c):
    """knn fill that provides the log probability"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = fancyimpute.KNN(k=3).fit_transform(df)
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return -scipy.stats.norm(drug_preds, std * 1).logpdf(drug_true).sum()


def knn_fill_conf(perc, c):
    """knn fill that provides the conf intervals"""
    clf = BayesianRidge()
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    design = fancyimpute.KNN(k=3).fit_transform(df)
    clf.fit(design, y)
    drug_preds, std = clf.predict(drug_vals, return_std=True)
    return sum(scipy.stats.norm(drug_preds, std * 1).pdf(drug_true) < 0.05)


def multi_imp(perc, c, m=10):
    """multiple imputation for log prob"""
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    drug_dists = []
    prob_sum = 0  # guassian probs sum
    for k in range(m):
        clf = BayesianRidge()
        design = fancyimpute.IterativeImputer(n_iter=10, sample_posterior=True, random_state=int(
            k * 243624) % 2**(32 - 1)).fit_transform(df)
        clf.fit(design, y)
        drug_preds, std = clf.predict(drug_vals, return_std=True)
        prob_sum += (1 / m) * scipy.stats.norm(drug_preds,
                                               std * 1).pdf(drug_true).prod()
    return -np.log(prob_sum)


def multi_imp_conf(perc, c, m=10):
    """multiple imputations for conf intervals"""
    df, y = glm_testing.create_missing(perc=perc, c=c)
    drug_vals, drug_true = glm_testing.test_drug(c=c)
    prob_sum = np.zeros(20)  # guassian probs sum
    for k in range(m):
        clf = BayesianRidge()
        design = fancyimpute.IterativeImputer(n_iter=10, sample_posterior=True, random_state=int(
            k * 243624) % 2**(32 - 1)).fit_transform(df)
        clf.fit(design, y)
        drug_preds, std = clf.predict(drug_vals, return_std=True)
        prob_sum += (1 / m) * scipy.stats.norm(drug_preds,
                                               std * 1).pdf(drug_true)
    return sum(prob_sum < 0.05)  # 95 percent
