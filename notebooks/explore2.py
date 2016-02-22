import sys, itertools
import numpy as np
import sklearn, time, math
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

if '../tools' not in sys.path:
    sys.path.append('../tools')
from helper_functions import * 

import warnings
warnings.filterwarnings('ignore')

num_clusters, exemplar = 8, 3
T = num_clusters * 2
t = (exemplar - (1 if exemplar % 2 == 0 else 0)) * 2 - 1
X, Y = load_all_fv(num_clusters, exemplar) #load_all_fv(num_clusters, exemplar)
C, Y = load_chroma_fv(13, 11)
X = np.concatenate((X, C), axis=1)

def log_range(lo, hi): return [10 ** i for i in range(lo, hi)]

def runIterative(X, Y, hypers, classifier):
    hypers = list(hypers)
    best_acc, best_params = -np.inf, None
    t = time.time()
    skf = StratifiedKFold(Y, n_folds=3, shuffle=True, random_state=1)
    for i, C in enumerate(hypers):
        svm = classifier(C)
        score = np.average(cross_val_score(svm, X, Y, cv=skf, n_jobs=-1))
        if score > best_acc:
            best_acc = score
            best_params = C
        print('\r{} best acc {:03g} hyper {}'.format(
                completion_bar(i + 1, len(hypers), width=20),
                best_acc, best_params), end='')
        sys.stdout.flush()
    t = time.time() - t
    print('\ndone in', t,'seconds')
    return best_acc, best_params

import multiprocessing
def get_score(tup):
    C, X, Y, classifier, data_change, skf = tup
    svm = classifier(C)
    mX = X
    if data_change: mX = data_change(mX, C)
    return np.average(cross_val_score(svm, mX, Y, cv=skf))
def best(scores, hypers):
    best = max(scores)
    return best, hypers[scores.index(best)]
def run(X, Y, hypers, classifier, data_change=None):
    hypers = list(hypers)
    best_acc, best_params = -np.inf, None
    skf = StratifiedKFold(Y, n_folds=3, shuffle=True, random_state=1)
    scores = []
    nproc = 40
    with multiprocessing.Pool(nproc) as p:
        def mkclosure(h): return h, X, Y, classifier, data_change, skf
        for argsls in chunks([mkclosure(h) for h in hypers], nproc):
            scores.extend(p.map(get_score, argsls))
            print('\r{} best acc {} params {}'.format(
                completion_bar(len(scores), len(hypers), width=20),
                *best(scores, hypers)))
            sys.stdout.flush()
    print()
    return best(scores, hypers)

df = ['ovo', 'ovr']
Cs = log_range(0, 10)
kernels = ['rbf', 'poly']
def make_svm(tup):
    C, k, d = tup
    if k == 'poly': return SVC(C=C, kernel=k, decision_function_shape=d, degree=10., coef0=1.)
    return SVC(C=C, kernel=k, decision_function_shape=d)

print('unmodified mfcc+chroma on SVM (hyper is C, kernel, ovo/ovr)')
run(X, Y, itertools.product(Cs, kernels, df), make_svm)

loss = ['hinge', 'squared_hinge']
penalty = ['l1', 'l2']
def make_lsvm(tup):
    C, l, p = tup
    return LinearSVC(C=C, loss=l, penalty=p)
print('unmodified mfcc+chroma on LinearSVM (hyper is C, kernel, ovo/ovr)')
run(X, Y, itertools.product(Cs, loss, penalty), make_lsvm)
