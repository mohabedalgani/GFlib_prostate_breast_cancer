'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import ptitprince as pt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.stats import ttest_ind

# create folder for graphs generated during the run
if not os.path.exists('images/'):
    os.makedirs('images/')
else:
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

from inipop import inipop
from genpop import genpop
from tipicalsvm import typicalsvm
from time import time

from dask.distributed import Client
client = Client(processes=False,
                threads_per_worker=4,
                n_workers=4,
                memory_limit='8GB'
               ) # dask client for parallel operations
print(client)
filterwarnings('ignore')
print('Running binary classification ...\n\n')

DATA_PATH = 'data/binary/'  # Dataset path for binary classification

print('Type the maximum length of the chromosome: ')
max_chromosome_length = int(input())  # the maximum total length of the chromosome

params = dict()
params['type'] = 'binary'  # problem type10
params['data'] = 'Breast_NCA.txt'  # path to data file
params['kernel'] = 'gf'  # rbf,linear,polynomial,gf
params['mutProb'] = 0.5  # mutation probability
params['crossProb'] = 0  # crossover probability
params['maxGen'] = 25  # max generation
params['popSize'] = 50  # population size
params['crossVal'] = 3  # number of cross validation slits
params['opList'] = ['Plus_s', 'Minus_s', 'Multi_s', 'Plus_v', 'Minus_v', 'x', 'y']  # Operators and operands
params['useDask'] = True # use distributed training of begged SVM classifiers instead of one classifier (for huge  datasets)
params['nEstimators'] = 8 # n estimators for parallel SVM computations

print(f'''Data Set : {DATA_PATH + params['data']}\n\n''')
kernels = ['poly', 'rbf', 'linear', 'gf']
totalMSE = dict()
totalP = dict()
totalT = dict()
for ker in kernels:
    totalMSE[ker] = list()
    totalP[ker] = list()
    totalT[ker] = list()

model_predictions = {
}
for i in range(3):
    for index, kernel in enumerate(kernels):
        params['kernel'] = kernel
        print(f'''SVM Kernel : {params['kernel']} \n''')
        if kernel == 'gf':
            print(f'''Max Generation : {params['maxGen']}\n''')
            print(f'''Population Size : {params['popSize']}\n''')
            print(f'''CrossOver Probability : {params['crossProb']}\n''')
            print(f'''Mutation Probability : {params['mutProb']}\n\n''')
            pop = inipop(params, max_chromosome_length)  # generate initial population
            try:
                mse, y_pred_proba, y_test = genpop(pop, params, i)  # get the best population from the initial one
                t_value,p_value=stats.ttest_rel(y_test,y_pred_proba)
                print('t_value=%.3f, p_value=%.3f' % (t_value, p_value))
            except:
                mse = genpop(pop, params, i)
        else:
            try:
                mse, y_pred_proba, y_test = typicalsvm(params)
                t_value,p_value=stats.ttest_rel(y_test,y_pred_proba)
                print('t_value=%.3f, p_value=%.3f' % (t_value, p_value))
            except:
                mse = typicalsvm(params)
        model_predictions[params['kernel']] = [y_pred_proba, y_test]
        totalMSE[kernel].append(mse)
        # T-Test for significance
        # stat, p = ttest_ind(y_test, y_pred_proba)
        
        totalP[kernel].append(p_value)
        totalT[kernel].append(t_value)
       # alpha = 0.05
       # if p >= alpha:
       #     print('probably same distros')
       # else:
       #     print('different distros')
        
        print('\n')

# Plotting roc auc curves for all algorithms
for (key, value) in model_predictions.items():
    y_pred_proba, y_test = value
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{key}: AUC=" + str(auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
plt.savefig('images/auc.png')



# Boxplot of errors for each kernel
plt.figure(figsize = (7, 7))
plt.boxplot([totalMSE['poly'], totalMSE['rbf'], totalMSE['linear'], totalMSE['gf']])
plt.xticks(np.arange(1, 5), kernels)
plt.title('MSE for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('Test Error Rate')
plt.ioff()
plt.savefig('images/mse.png')
plt.show()

# codes for the pvalues and t-test results

# Boxplot of p_values for each kernel
plt.figure(figsize = (7, 7))
plt.boxplot([totalP['poly'], totalP['rbf'], totalP['linear'], totalP['gf']])
plt.xticks(np.arange(1, 5), kernels)
plt.title('P values for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('P values Rate')
plt.ioff()
plt.savefig('images/pValues.png')
plt.show()


# Boxplot of t_values for each kernel
plt.figure(figsize = (7, 7))
plt.boxplot([totalT['poly'], totalT['rbf'], totalT['linear'], totalT['gf']])
plt.xticks(np.arange(1, 5), kernels)
plt.title('T values for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('T values Rate')
plt.ioff()
plt.savefig('images/tValues.png')
plt.show()

#Raincloud plot
#ax = pt.RainCloud(x = '', y = ,
#                  data = totalMSE,
#                  width_viol = .8,
#                  width_box = .4,
#                  orient = 'h',
#                  move = .0)
#plt.savefig('images/raincloud.png')

