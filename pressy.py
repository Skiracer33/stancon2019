import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import BayesianRidge 
import joblib 
import os
import censor_fix


def imputation_ex():
    df = pd.DataFrame({'a': {0: 0.589,
                             1: 0.922,
                             2: 0.182,
                             4: 0.186},
                       'b': {0: 0.638,
                             1: 0.900,
                             3: 0.838,
                             4: 0.755}}
                      )
    df[['a_imputed', 'b_imputed']] = df.fillna(
        value=0.5)  # performing the imputation
    df = df.rename_axis('drug')
    return df


def complete_case_ex():
    plt.title('Complete Case Analysis')
    missing_data = np.arange(0, 40, 0.1)
    plt.plot(missing_data, (1 - missing_data / 100)**3 * 100, label='3')
    plt.plot(missing_data, (1 - missing_data / 100)**9 * 100, label='9')
    plt.plot(missing_data, (1 - missing_data / 100)**27 * 100, label='27')
    plt.legend(title='Columns')
    plt.ylim(0, 100)
    plt.xlabel('Missing Data Per Column (%)')
    plt.ylabel('Rows Remaining for \n complete case analysis (%)')


def plot_surprisal(title, x):
    d = x.mean(axis=2)
    x_axis = np.arange(4, 52, 4)
    y_axis = np.linspace(0.1, 0.9, 9)
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.title(title)
    plt.rc('font', size=22)
    plt.xlabel('Missing Data (%)')
    plt.ylabel('- Log Probability')
    for i in range(0, 9, 2):
        plt.plot(x_axis, (d[:, i]))
    plt.legend([round(y_axis[i], 2)
                for i in range(0, 9, 2)], title='Correlation')


def plot_all_surprisal():
    fig, axs = plt.subplots(5, 3, figsize=(10, 10), sharey=True, sharex=True)
    fig.suptitle('Comparison of Imputation Surprisal')
    fig.text(0.5, 0.04, 'Missing Data (%)', ha='center')
    fig.text(0.02, 0.5, '- Log Prob', va='center', rotation='vertical')
    fig.text(0.96, 0.5, 'Correlation', va='center', rotation='vertical')

    for i in range(3):
        axs[0, i].set_title(['Mean', 'KNN', 'Multi Regress'][i])

    for j in range(0, 3):
        x = joblib.load(names[j])
        d = x.mean(axis=2)
        x_axis = np.arange(4, 52, 4)
        y_axis = np.linspace(0.1, 0.9, 9)
        for v, i in enumerate(range(0, 9, 2)):
            colour = ['tab:orange', 'tab:red', 'tab:green'][j]
            axs[v, j].plot(x_axis, d[:, i], colour, label=str(i))

    for i, ax in enumerate(axs.flat):
        x_lab = ['Mean', 'KNN', 'Multi'][i % 3]
        if i % 3 == 2:
            ax.set(ylabel='c = {:.1f}'.format(y_axis[2 * (i - 2) // 3]))
            ax.yaxis.set_label_position("right")


def plot_coverage(title, x):
    d = x.mean(axis=2)
    x_axis = np.arange(4, 52, 4)
    y_axis = np.linspace(0.1, 0.9, 9)
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.title(title)
    plt.rc('font', size=22)
    plt.xlabel('Missing Data (%)')
    plt.ylabel('Coverage (%)')
    for i in range(0, 9, 2):
        plt.plot(x_axis, (100 - 5 * d[:, i]))
    plt.legend([round(y_axis[i], 2)
                for i in range(0, 9, 2)], title='Correlation')


def plot_all_coverage():
    fig, axs = plt.subplots(5, 3, figsize=(10, 10), sharey=True, sharex=True)
    fig.suptitle('Comparison of Imputation Coverage')
    fig.text(0.5, 0.04, 'Missing Data (%)', ha='center')
    fig.text(0.02, 0.5, 'Coverage (%)', va='center', rotation='vertical')
    fig.text(0.96, 0.5, 'Correlation', va='center', rotation='vertical')

    for i in range(3):
        axs[0, i].set_title(['Mean', 'KNN', 'Multi'][i])

    for j in range(3, 6):
        x = joblib.load(names[j])
        d = x.mean(axis=2)
        x_axis = np.arange(4, 52, 4)
        y_axis = np.linspace(0.1, 0.9, 9)

        for v, i in enumerate(range(0, 9, 2)):
            colour = ['tab:orange', 'tab:red', 'tab:green'][j - 3]
            axs[v, j - 3].plot(x_axis, 100 - d[:, i] * 5, colour, label=str(i))

    for i, ax in enumerate(axs.flat):
        x_lab = ['Mean', 'KNN', 'Multi'][i % 3]
        if i % 3 == 2:
            ax.set(ylabel='c = {:.1f}'.format(y_axis[2 * (i - 2) // 3]))
            ax.yaxis.set_label_position("right")


def censor_ex():
    plt.rc('font', size=16)
    x = np.random.rand(100)
    y = x**3 + 2 * x**0.1 + 0.5 * np.random.rand(100)
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.title('Drug Data True Values')
    plt.xlabel('BSEP')
    plt.ylabel('SPHER')
    plt.ylim(1, 3.5)

    y[y > 2.3] = 2.3
    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.title('Drug Data Censored')
    plt.xlabel('BSEP')
    plt.ylabel('SPHER')
    plt.ylim(1, 3.5)


def censor_fix_ex():
    plt.rc('font', size=12)
    fig = plt.figure(figsize=(15, 5))
    from censor_fix import censorfix
    imp = censorfix.censorImputer(
        debug=False, no_columns=1, sample_posterior=False,)
    df = pd.DataFrame([y, x]).T
    df = df.sort_values(by=0, ascending=True)
    imp.impute_once(df[0], df[[1]], 2.3, 'NA')

    plt.subplot(1, 2, 1)
    plt.scatter(df.iloc[:, 1], df.iloc[:, 0])
    plt.title('Drug Data Best Imputation')
    plt.xlabel('BSEP')
    plt.ylabel('SPHER')
    plt.ylim(1, 3.5)

    imp = censorfix.censorImputer(
        debug=False,
        no_columns=1,
        sample_posterior=True)
    df = pd.DataFrame([y, x]).T
    df = df.sort_values(by=0, ascending=True)
    imp.impute_once(df[0], df[[1]], 2.3, 'NA')
    plt.subplot(1, 2, 2)

    plt.scatter(df.iloc[:, 1], df.iloc[:, 0])
    plt.title('Drug Data Imputation from the Bayesian posterior ')
    plt.xlabel('BSEP')
    plt.ylabel('SPHER')
    plt.ylim(1, 3.5)


def create_data():
    c = 0.5
    n = 3
    cov = c + np.identity(n) * (1 - c)
    size = 100
    full_data = np.random.multivariate_normal(
        [0 for i in range(n)], cov, size=size)
    df = pd.DataFrame(full_data)
    df2 = df.copy()
    return df, df2


def censor_fix2d_ex():
    import censor_fix
    df, df2 = create_data()
    df.loc[df[0] > 0.8, 0] = 0.8  # applying censoring
    df.loc[df[0] < -0.8, 0] = -0.8
    df.loc[df[1] > 0.8, 1] = 0.8
    df.loc[df[1] < -0.8, 1] = -0.8
    imp = censor_fix.censorfix.censorImputer(
        debug=False, sample_posterior=True)
    U = [0.8, 0.8, 'NA']  # the upper censor values
    L = [-0.8, -0.8, 'NA']  # the lower censor values

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    df.plot(kind='scatter', x=0, y=1, ax=ax, color='yellow', label='censored')
    df = imp.impute(df, U, L, iter_val=5)
    df2.plot(
        kind='scatter',
        x=0,
        y=1,
        ax=ax,
        color='pink',
        label='imputed_values')
    df.plot(kind='scatter', x=0, y=1, ax=ax, label='actual')
    plt.legend()
    plt.title('Multivariate Censor Imputation')


def final_comparison():
    def run_analysis(y, X):  # Analysis
        clf = LogisticRegression(solver='lbfgs', max_iter=5000)
        X_train, X_test = train_test_split(X)
        scores = []
        X_train = pf.fit_transform(X_train)
        clf.fit(X_train, y_train)
        X_test = pf.transform(X_test)
        y_pred = clf.predict(X_test)
        return accuracy_scores(y_pred, y_test)

    y, X = joblib.load('dili_data.pkl')  # Load data
    imp = censor_fix.censorfix.censorImputer(debug=False, sample_posterior=True,
                                             n_jobs=16, distribution='gaussian')  # Specify imputation methods
    U = X.apply(max)  # Specify the censoring points
    L = ['NA'] * X.shape[1]  # Specify the censoring points
    # create 100 imputations
    imp_X = imp.impute(X, U, L, iter_val=2, no_imputations=100)

    scores = []
    for imp_data in imp_X:
        # Run the 100 imputations through the analysis
        scores.append(run_analysis(y, imp_data))
        print(np.mean(scores))  # The average accuracy
        print(np.std(scores))  # How much the imputation affects the accuracy