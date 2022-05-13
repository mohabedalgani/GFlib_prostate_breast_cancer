'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, mean_squared_error

from kernel import Kernel

from sklearn.decomposition import PCA
import time

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import joblib
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.stats import mode
from warnings import filterwarnings; filterwarnings('ignore')

PATH = 'data/'
rnd_seed = 2019
MAX_ITER = 100
DASK_SAMPLE_SIZE = 30
N_COMPONENTS = 10

useSamples = 10000


def train_model(X, y, model, ests):
    '''
    For parallel training
    :param X:
    :param y:
    :param model:
    :return:
    '''
    sample = np.random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * 1 / ests), replace=False)
    X_, y_ = X[sample], y[sample]
    return model.fit(X_, y_)


def predict_model(X, model):
    '''
    For parallel prediction
    :param X:
    :param model:
    :return:
    '''
    return model.predict(X)


def calcfitness(pop, params):
    '''
    Reads the data of type params['type'] and with the data path params['data'],
    then fits the SVC or SVR model depending on the params['type'] with the custom
    kernel, determined by the "pop" parameter. Calculates the resulting metrics for the
    input population.

    :param pop: Population, which will determine the custom kernel for SVM model
    :param params: Parameters, containing the info about population and about the task we are solving
    :return: -MSE for regression task, Accuracy * 100 for the binary and multi classification tasks
    '''

    models = []
    fitness = []
    if params['type'] == 'binary':
        if params['data'] == 'ASTD.txt': #spam.txt
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                #tmpX = M[:, :-1]
                #tmpY = M[:, -1]
                tmpX = M[:, -1]
                tmpY = M[:, 0]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
        elif params['data'] == 'breast-cancer-wisconsin.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = []
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    else:
                        M.append(tmp)
                M = np.asarray(M)
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]

        elif params['data'] == 'wdbc.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = []
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    else:
                        M.append(tmp)
                M = np.asarray(M)
                tmpX = np.delete(M, 1, 1) # leave all columns except the 1st one
                tmpY = M[:, 1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]

        elif params['data'] == 'network-intrution.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = []
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    else:
                        M.append(tmp)
                M = np.asarray(M)
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
        elif params['data'] == 'german credit.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
# Friday protate cancer 2021-2022
#https://www.kaggle.com/willribeiro/logistic-regression-classifier-prostate-cancer/notebook
#https://www.kaggle.com/willribeiro/logistic-regression-classifier-prostate-cancer/data
    # 4 models to compare with
    # https://www.kaggle.com/smogomes/prostate-cancer-prediction-model
        elif params['data'] == 'Prostate_Cancer_scaled.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
#End prostate cancer        
        
#https://www.kaggle.com/mysarahmadbhat/lung-cancer
# scaled
        elif params['data'] == 'lung_Cancer_Scaled.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
#End lung cancer
#https://www.kaggle.com/mozmezher/breast-cancer-eda-prediction-99-acc/data?scriptVersionId=88169397
# scaled
        elif params['data'] == 'BreastCancerKaggle.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
#End Breast Cancer Kaggle
##https://www.kaggle.com/code/mozmezher/breast-cancer-eda-prediction-99-acc/edit/run/88169397
# scaled
        elif params['data'] == 'Breast_NCA.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
#End Breast Cancer Kaggle
        elif (params['data'] == 'logic_6_multiplexer.txt') | (params['data'] == 'odd_7_parity.txt') | (params['data'] == 'odd_3_parity.txt'):
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
        elif params['data'] == 'credit approval.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
            M = pd.DataFrame(M)
            M = M[~(M == '?').any(axis=1)]
            for col in M.columns:
                try:
                    M.loc[:, col] = M[col].map(float)
                except:
                    M.loc[:, col] = LabelEncoder().fit_transform(M[col])
            M = M.values
            tmpX = M[:, :-1]
            tmpY = M[:, -1]

            tmpX = tmpX[:useSamples]
            tmpY = tmpY[:useSamples]
        else:
            with open(PATH + 'binary/' + params['data']) as f:
                lenConst = 0
                if params['data'] == 'sonar_scale.txt':
                    lenConst = 61
                if params['data'] == 'ionosphere_scale.txt':
                    lenConst = 34
                if params['data'] == 'heart_scale.txt':
                    lenConst = 13
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(':'))
                    tmpp = []
                    for t in tmp:
                        tmpp.append(t.split(' ')[0])
                    if len(tmpp) != lenConst:
                        continue
                    if M.shape[0] == 0:
                        M = np.array(tmpp)
                    else:
                        M = np.vstack([M, tmpp])

                #tmpX = M[:, 1:]
                #tmpY = M[:, 0]
                tmpX = M[:, -1]
                tmpY = M[:, 0]
                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]

        tmpX = pd.DataFrame(tmpX)

        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])

        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size = .75, random_state=rnd_seed)

        if trainX.shape[0] < DASK_SAMPLE_SIZE*params['nEstimators']:
            params['useDask'] = False
            print('Dask turned off due to not sufficient number of samples')

        for i in range(params['popSize']):
            ind = pop[i]  # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)
            if params['useDask']:
                # use k svms with data sample in parallel
                svms = Parallel(n_jobs=mp.cpu_count())(delayed(train_model)(trainX, trainY, SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True),params['nEstimators']) for est in range(params['nEstimators']))
                 # choose most common prediction for the svm ensemble
                label = mode(np.array(Parallel(n_jobs=mp.cpu_count())(delayed(predict_model)(testX, svm) for svm in svms)), axis = 0)[0].reshape(testX.shape[0])
                models.append(svms[-1])
            else:
                svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)  # create an SVM model with custom kernel
                svm.fit(trainX, trainY)
                label = svm.predict(testX)
                models.append(svm)
            fitness.append(accuracy_score(testY, label) * 100)

    if params['type'] == 'multi':
        if params['data'] == 'zoo.txt':
            with open(PATH + 'multi/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]
        else:
            with open(PATH + 'multi/' + params['data']) as f:
                if params['data'] == 'wine_scale.txt':
                    lenConst = 14
                if params['data'] == 'iris_scale.txt':
                    lenConst = 5
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(':'))
                    tmpp = []
                    for t in tmp:
                        tmpp.append(t.split(' ')[0])
                    if len(tmpp) != lenConst:
                        continue
                    if M.shape[0] == 0:
                        M = np.array(tmpp)
                    else:
                        M = np.vstack([M, tmpp])

                tmpX = M[:, 1:]
                tmpY = M[:, 0]

                tmpX = tmpX[:useSamples]
                tmpY = tmpY[:useSamples]

        tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75, random_state=rnd_seed)

        if trainX.shape[0] < DASK_SAMPLE_SIZE*params['nEstimators']:
            params['useDask'] = False
            print('Dask turned off due to not sufficient number of samples')

        for i in range(params['popSize']):
            ind = pop[i]  # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)

            if params['useDask']:
                svms = Parallel(n_jobs=mp.cpu_count())(delayed(train_model)(trainX, trainY, SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)) for est in range(params['nEstimators']))
                label = mode(np.array(Parallel(n_jobs=mp.cpu_count())(delayed(predict_model)(testX, svm) for svm in svms)), axis = 0)[0].reshape(testX.shape[0])
                models.append(svms[-1])
            else:
                svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)  # create an SVM model with custom kernel
                svm.fit(trainX, trainY)
                label = svm.predict(testX)
                models.append(svm)
            fitness.append(accuracy_score(testY, label) * 100)
    if params['type'] == 'regress':
        with open(PATH + 'regress/'+ params['data']) as f:
            if params['data'] == 'abalone_scale.txt':
                lenConst = 9
            if params['data'] == 'housing_scale.txt':
                lenConst = 14
            if params['data'] == 'mpg_scale.txt':
                lenConst = 8
            M = np.array([])
            file = f.read().split('\n')
            for val in file:
                tmp = np.array(val.split(':'))
                tmpp = []
                for t in tmp:
                    tmpp.append(t.split(' ')[0])
                if len(tmpp) != lenConst:
                    continue
                if M.shape[0] == 0:
                    M = np.array(tmpp)
                else:
                    M = np.vstack([M, tmpp])

            tmpX = M[:, 1:]
            tmpY = M[:, 0]

            tmpX = tmpX[:useSamples]
            tmpY = tmpY[:useSamples]

        tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75, random_state=rnd_seed)

        if trainX.shape[0] < DASK_SAMPLE_SIZE*params['nEstimators']:
            params['useDask'] = False
            print('Dask turned off due to not sufficient number of samples')

        for i in range(params['popSize']):
            ind = pop[i]   # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)

            if params['useDask']:
                svms = Parallel(n_jobs=mp.cpu_count())(delayed(train_model)(trainX, trainY, SVR(max_iter=MAX_ITER, kernel=k.kernel)) for est in range(params['nEstimators']))
                label = mode(np.array(Parallel(n_jobs=mp.cpu_count())(delayed(predict_model)(testX, svm) for svm in svms)), axis = 0)[0].reshape(testX.shape[0])
                models.append(svms[-1])
            else:
                svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)  # create an SVM model with custom kernel
                svm.fit(trainX, trainY)
                label = svm.predict(testX)
                models.append(svm)
            fitness.append(-mean_squared_error(testY, label))
    return fitness, models, testX, testY
