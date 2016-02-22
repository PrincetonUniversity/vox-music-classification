import sys, itertools
import numpy as np
import sklearn, time, math
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

if '../tools' not in sys.path:
    sys.path.append('../tools')
from helper_functions import * 

import warnings
warnings.filterwarnings('ignore')

num_clusters_mfcc = sys.argv[1]
exemplar_size_mfcc = sys.argv[2]
num_clusters_chroma = sys.argv[3]
exemplar_size_chroma = sys.argv[4]

def make_rf(md):
    return RandomForestClassifier(n_estimators=(max(md,10)*6), max_depth=md)

def logRange(lo, hi): return [10 ** i for i in range(lo, hi)]
Cs = logRange(0, 8)

def run(X, Y, hypers, classifier, data_change=None):
    hypers = list(hypers)
    best_acc, best_params = -np.inf, None
    skf = StratifiedKFold(Y, n_folds=3, shuffle=True, random_state=1)
    for i, C in enumerate(hypers):
        svm = classifier(C)
        mX = X
        if data_change: mX = data_change(mX, C)
        score = np.average(cross_val_score(svm, mX, Y, cv=skf))
        if score > best_acc:
            best_acc = score
            best_params = C
    return best_acc, best_params

def order(x):
    p = [x[0] for x in sorted(list(enumerate(x)), key=lambda tup:tup[1])]
    p.reverse()
    return p

rf = RandomForestClassifier(n_estimators=200, max_depth=30)

best_acc, best_c = -np.inf, 1
t = time.time()



X, Y = load_all_fv(num_clusters_mfcc, exemplar_size_mfcc)
C, Y = load_all_fv(num_clusters_chroma, exemplar_size_chroma)
X = np.concatenate((X, C), axis=1)

skf = StratifiedKFold(Y, n_folds=3, shuffle=True, random_state=1)
overall_score = 0
for train, test in skf:
    trX, teX = X[train], X[test]
    trY, teY = Y[train], Y[test]
    depth = run(trX, trY, [10, 20, 30, 40, 50], make_rf)[1]
    rf = make_rf(depth)
    rf.fit(trX, trY)
    p = order(rf.feature_importances_)

    hypers = itertools.product(
        itertools.chain([len(p)], range(500, len(p), 500)),
        logRange(0, 7))
    limit, C = run(trX, trY, hypers, lambda t: SVC(C=t[1]),
                   lambda X, t: X[:, p[:t[0]]])[1]

    trX, teX = trX[:, p[:limit]], teX[:, p[:limit]]
    svm = SVC(C=C)
    svm.fit(trX, trY)
    score = svm.score(teX, teY)
    print('rf-depth {} limit {} C {} score {}'.format(
        depth, limit, C, score))
    overall_score += score
print('avg cv {}'.format(overall_score / 3))
