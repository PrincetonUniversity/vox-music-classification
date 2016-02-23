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
    with multiprocessing.Pool(8) as p:
        return p.map(rfit, sets)

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

num_clusters = int(sys.argv[1])
exemplar = int(sys.argv[2])

print('printing rf data for {} clusters {} exemplar'.format(
    num_clusters, exemplar))

M, Y = load_all_fv(num_clusters, exemplar)
C, Y = load_chroma_fv(num_clusters, exemplar)
X = np.concatenate((M, C), axis=1)
cv = cvFolds(X, Y)
def run(prop): return run_rf(cv, prop)
with multiprocessing.Pool(20) as p:
    print(p.map(run, np.arange(0.01, 1, 0.05)))
