import sys, itertools
import numpy as np
import sklearn, time, math
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

if '../tools' not in sys.path:
    sys.path.append('../tools')
from helper_functions import * 

import warnings
warnings.filterwarnings('ignore')
def log_range(lo, hi): return [10 ** i for i in range(lo, hi)]
    
def best(scores, hypers):
    best = max(scores)
    return best, hypers[scores.index(best)]

import multiprocessing
def make_svm(C):
    return SVC(C=C, kernel='poly',
               decision_function_shape='ovo',
               degree=10., coef0=1.)
Cs = log_range(0, 15)
def rf(): return RandomForestClassifier(n_estimators=200, max_depth=20)
def rfit(tup):
    trX, teX, trY, teY = tup
    return order(rf().fit(trX, trY).feature_importances_), trX, teX, trY, teY
def cvFolds(X, Y):
    skf = StratifiedKFold(Y, n_folds=8, shuffle=True,
                          random_state=1)
    sets = [(X[train], X[test], Y[train], Y[test]) for train, test in skf]
    return [rfit(t) for t in sets]

def eval_fold(tup):
    (prop, C), (p, trX, teX, trY, teY) = tup
    svm = make_svm(C)
    p = p[:int(len(p) * prop)]
    trXp, teXp = trX[:, p], teX[:, p]
    return svm.fit(trXp, trY).score(teXp, teY)
    
def run_rf(folds, prop):
    best_acc, best_params = -np.inf, None
    #trX, teX = whiten(trX), whiten(teX, source=trX)
    for i, C in enumerate(Cs):
        scores = [eval_fold(x) for x in zip(itertools.repeat((prop, C)), folds)]
        score = np.average(scores)
        if score > best_acc:
            best_acc = score
            best_params = C
    return prop, best_acc

def to_try(x): return [(8,3)] + list((x, x) for x in range(3, 12)) + list((x, 11) for x in range(3, x))
l = list(itertools.product(to_try(21), to_try(31)))
assert(len(l) % 4 == 0)

machine = int(sys.argv[1])

l = list(chunks(l, len(l) // 4))[machine]

print('Running params for machine {}:\n{}'.format(machine, l))

def try_config(params):
    print('starting', params)
    mfcc, chroma = params
    M, Y = load_all_fv(*mfcc)
    C, Y = load_chroma_fv(*chroma)
    X = np.concatenate((M, C), axis=1)
    cv = cvFolds(X, Y)

    best_acc = -np.inf
    best_prop = 0
    strikes = 0
    for prop in np.arange(0.01, 1, 0.05):
        acc = run_rf(cv, prop)[1]
        if acc > best_acc:
            best_acc = acc
            best_prop = prop
            strikes = 0
        else:
            strikes += 1
            if strikes == 2: break
    print('finished', params, 'acc', best_acc, 'prop', best_prop)

# should be run on cycles
with multiprocessing.Pool(48) as p:
    p.map(try_config, l)
