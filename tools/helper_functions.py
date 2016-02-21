"""
This module contains assorted helper functions used in analysis, to be located
in one place in order to avoid clutter and duplication in jupyter notebooks.
"""
import subprocess
import scipy.io as spio
import numpy as np
from os.path import join, isdir, isfile
import sys, os

def load_mfcc_labels(num_clusters, exemplar_size):
    """Returns a tuple mfcc, labels of the mfcc data and labels for the songs
       given the cluster and exemplar parameters. """
    FVs = '../generated-fv/FV' + str(num_clusters) + '-' + str(exemplar_size) + '.mat'
    LBs = '../generated-fv/LB.mat'
    assert isdir('../generated-fv') # Should be running in a project root subdir
    if not isfile(FVs) or not isfile(LBs):
        print('Generating Fisher Vectors {} clusters {} exemplar'.format(
            num_clusters, exemplar_size))
        cmd = 'matlab -nodisplay -nosplash -nodesktop -r '
        cmd += 'addpath(\'../tools\');FV_concat({},{});exit;'
        cmd = cmd.format(num_clusters, exemplar_size)
        print('cmd:', cmd.split())
        sys.stdout.flush()
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        print('MATLAB output:\n' + process.communicate()[0].decode('utf-8'))
    else:
        print('Using existing FVs from file {}'.format(FVs))
    mfcc = np.transpose(spio.loadmat(FVs)['FV'])
    labels = spio.loadmat(LBs)['LB'][0]
    return mfcc, labels

# From Stack Overflow
# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def load_all_fv(num_clusters, exemplar_size):
    """Loads the FV and runs basic validation, checking there are only
    10 labels total. Does a shuffle."""
    mfcc, labels = load_mfcc_labels(num_clusters, exemplar_size)

    N = len(mfcc)
    nlabels = len(set(labels))
    assert nlabels == 10
    per_label = N // nlabels
    for chunk in chunks(labels, per_label):
        assert(len(set(chunk)) == 1)
    print('N = {}'.format(N))

    def summary(x): return '[{:.4f}, {:.4f}]'.format(x.min(), x.max())
    print('MFCC training feature ranges means {} sds {}'.format(
        summary(np.mean(mfcc, axis=0)),
        summary(np.std(mfcc, axis=0))))
    return mfcc, labels

def whiten(x, source=None):
    """Mean and sd normalizes x column-wise relative to summary statistics
    from source. Uses x itself by default"""
    if source is None: source = x
    means = np.mean(source, axis=0)
    stddevs = np.std(source, axis=0)
    return (x - means[None, :]) / stddevs[None, :]

def riffle_perm(N, t, T):
    """Creates a riffle permutation, which, when applied to an array x,
    riffles the data, where for every interval of size T in x,
    the columns are re-shuffled such that columns with index in the
    same equivalence class modulo t are adjacent, in the same order
    within classes and between the first members of each class."""
    assert N % T == 0
    assert (N // T) % t == 0
    perm = np.empty(N, dtype=int)
    ctr = 0
    for i in range(0, N, N // T):
        for j in range(N // T // t):
            nxt = ctr + t
            perm[ctr:nxt] = range(j, N // T, N // T // t)
            perm[ctr:nxt] += i
            ctr = nxt
    assert ctr == N
    return perm

def riffle(X, t, T):
    """Riffles a data set (i.e., each row).
    Reshapes X to Nx(D // t)xt, where N,D = X.shape"""
    N, D = X.shape
    perm = riffle_perm(D, t, T)
    return X[:, perm].reshape(N, D // t, t)

def completion_bar(position, total, width=50):
    frac_done = width * position // total
    return ('[' + '-' * frac_done + ' ' * (width - frac_done)
            + '] {:0' + str(len(str(total)))
            + 'd}/{}').format(position, total)

def pad(nparrs):
    shape = tuple((max(c) for c in zip(*(x.shape for x in nparrs))))
    def to_pad(x): return tuple(((0, a - b) for a, b in zip(shape, x.shape)))
    return [np.pad(x, to_pad(x), mode='constant') for x in nparrs]

def load_all_nonmfc():
    """Loads all features except for MFCC in the same order that MFCC
    is loaded"""

    ## List of all features that can be extracted ##
    L = ['eng', 'chroma', 't', 'keystrength', 'brightness', 'zerocross',
         'roughness', 'inharmonic', 'hcdf']

    ## Dictionary of feature matrices ##
    D = {}

    for feature in L:
        X = []
        for genre in sorted(os.listdir('../data')):
            path = join('../data', genre)
            if not isdir(path): continue
            def load(base, feat):
                return spio.loadmat(join(path, base))['DAT'][feat][0,0]
            arrs = [load(i, feature) for i in sorted(os.listdir(path))]
            X.append(np.array(pad(arrs)))
        print('Read in', feature, 'for all genres')
        X = np.concatenate(pad(X))
        D[feature] = X
