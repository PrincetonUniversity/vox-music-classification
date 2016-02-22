import sys, itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn, time, math
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

if '../tools' not in sys.path:
    sys.path.append('../tools')
from helper_functions import * 

import warnings
warnings.filterwarnings('ignore')

X, Y = load_all_fv(1,1)

def logRange(lo, hi): return [10 ** i for i in range(lo, hi)]
Cs = logRange(-1, 10)
kernels = ['rbf']
hypers = list(itertools.product(Cs, kernels))

def print_svc(C, k): return 'C={:03g} kernel={}'.format(C, k)
def make_svc(C, k):
    if k == 'poly': return SVC(C=C, kernel=k, coef0=1.0, degree=5)
    else: return SVC(C=C, kernel=k)

best_acc, best_params = -np.inf, None
t = time.time()
skf = StratifiedKFold(Y, n_folds=4, shuffle=True, random_state=1)
for clusters, exemplar in itertools.product(range(1, 11), range(1, 11)):
    X, Y = load_all_fv(clusters, exemplar)
    T = clusters * 2
    tt = (exemplar - (1 if exemplar % 2 == 0 else 0)) * 2 - 1
    #X = np.average(riffle(X, tt, T), axis=2)
    
    print('num clusters {} exemplar size {}'.format(clusters, exemplar))
    
    best_in_data_acc, best_in_data_params = -np.inf, None
    for i, tup in enumerate(hypers):
        svm = make_svc(*tup)
        score = np.average(cross_val_score(svm, X, Y, cv=skf,
                                           n_jobs=-1, pre_dispatch='2*n_jobs'))
        if score > best_in_data_acc:
            best_in_data_acc = score
            best_in_data_params = tup
        print('\r{} acc {:03g} param {}'.format(
            completion_bar(i + 1, len(hypers), width=1), best_in_data_acc,
            print_svc(*best_in_data_params)), end='')
        sys.stdout.flush()
    
    if best_in_data_acc > best_acc:
        best_acc = best_in_data_acc
        best_params = clusters, exemplar, \
            best_in_data_params[0], best_in_data_params[1]
    
    print('\nFinished data instance in {}s. Top performer:'.format(
        time.time() - t))
    print('  cv acc {} clusters {} exemplar {} C {} kernel {}\n'.format(
        best_acc, *best_params))
t = time.time() - t

print('Done with CV for {}-size hyper grid in {:03f}s'.format(
        len(hypers) * 100, t))
#print('Accuracies: best avg cv {} for {}'.format(best_acc, best_params)))
# (4, 5) (cv is 0.667) .76
# (6, 6) .76
