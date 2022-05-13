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

from dask.distributed import Client
client = Client(processes=False,
                threads_per_worker=4,
                n_workers=1,
                memory_limit='4GB'
               ) # dask client for parallel operations

filterwarnings('ignore')
print('Running regression ...\n\n')

print('Type the maximum length of the chromosome: ')
max_chromosome_length = int(input())  # the maximum total length of the chromosome

DATA_PATH = 'data/regress/'  # Dataset path for binary classification

params = dict()
params['type'] = 'regress'  # problem type
params['data'] = 'mpg_scale.txt'  # path to data file
params['kernel'] = 'gf'  # rbf,linear,polynomial,gf
params['mutProb'] = 0.1  # mutation probability
params['crossProb'] = 0.5  # crossover probability
params['maxGen'] = 2  # max generation
params['popSize'] = 5  # population size
params['crossVal'] = 5  # number of cross validation slits
params['opList'] = ['Plus_s', 'Minus_s', 'Plus_v', 'Minus_v',
                    'Sine', 'Cosine', 'Tanh', 'Log', 'x', 'y']  # Operators and operands
params['useDask'] = True # use distributed training of begged SVM classifiers instead of one classifier (for huge  datasets)
params['nEstimators'] = 10 # n estimators for parallel SVM computations

print(f'''Data Set : {DATA_PATH + params['data']}\n\n''')
kernels = ['poly', 'rbf', 'linear', 'gf']
totalMSE = dict()
for ker in kernels:
    totalMSE[ker] = list()

for i in range(5):
    temp = []
    for index, kernel in enumerate(kernels):
        params['kernel'] = kernel
        print(f'''SVM Kernel : {params['kernel']} \n''')
        if kernel == 'gf':
            print(f'''Max Generation : {params['maxGen']}\n''')
            print(f'''Population Size : {params['popSize']}\n''')
            print(f'''CrossOver Probability : {params['crossProb']}\n''')
            print(f'''Mutation Probability : {params['mutProb']}\n\n''')
            pop = inipop(params, max_chromosome_length)
            mse = genpop(pop, params, i)
        else:
            mse = typicalsvm(params)
        totalMSE[kernel].append(mse)
        print('\n')


# Boxplot of errors for each kernel
plt.boxplot([totalMSE['poly'], totalMSE['rbf'], totalMSE['linear'], totalMSE['gf']])
plt.xticks(np.arange(1, 5), kernels)
plt.title('MSE for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('Test Error Rate')
plt.ioff()
plt.savefig('images/mse.png')
plt.show()
