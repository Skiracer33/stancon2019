import sys
import joblib
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
import importlib
importlib.reload(workers)

pool = Pool(processes=12)

directory = sys.argv[3]


def test_grid(
    callback, runs=int(
        sys.argv[2]), x_axis=np.arange(
            4, 52, 4), y_axis=np.linspace(
                0.1, 0.9, 9)):
    res_grid = np.zeros((len(x_axis), len(y_axis), runs))
    for i, perc in tqdm(enumerate(x_axis)):
        for j, c in enumerate(y_axis):
            if __name__ == '__main__':
                res = [pool.apply_async(callback, [perc, c])
                       for z in range(runs)]
                for z in range(runs):
                    res_grid[i, j, z] = res[z].get()
    return res_grid  # ,log_like2,log_like3

# compute the mean fill knn fill and multi fill


if sys.argv[1] not in '012345':
    print('Error {} not a valid command'.format(sys.argv[1]))

if sys.argv[1] == '0':
    mf_bay = test_grid(workers.mean_fill)
    joblib.dump(mf_bay, directory + '/mf_bay')

if sys.argv[1] == '1':
    knn_bay = test_grid(workers.knn_fill)
    joblib.dump(knn_bay, directory + '/knn_bay')

if sys.argv[1] == '2':
    multi_bay = test_grid(workers.multi_imp)
    joblib.dump(multi_bay, directory + '/multi_bay')

if sys.argv[1] == '3':
    mf_bay_conf = test_grid(workers.mean_fill_conf)
    joblib.dump(mf_bay_conf, directory + '/mf_bay_conf')

if sys.argv[1] == '4':
    knn_bay_conf = test_grid(workers.knn_fill_conf)
    joblib.dump(knn_bay_conf, directory + '/knn_bay_conf')

if sys.argv[1] == '5':
    multi_bay_conf = test_grid(workers.multi_imp_conf)
    joblib.dump(multi_bay_conf, directory + '/multi_bay_conf')
