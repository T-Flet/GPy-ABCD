import numpy as np
import pandas as pd
from plotnine import ggplot, geom_line, aes


def sortOutTypePair(k1, k2):
    t1, t2 = type(k1), type(k2)
    return {t1: [k1, k2]} if t1 == t2 else {t1: k1, t2: k2}


def sampleCurves(k, xlims = (-3., 3.), n_curves = 5, same_kernel = False):
    '''Plot sample curves from a given kernel'''
    X = np.linspace(*xlims, 500)[:, None] # column vector of evenly spaced points
    mu = np.zeros(500) # vector of the means
    if same_kernel:
        k.randomize()
        Z = np.random.multivariate_normal(mu, k.K(X, X), n_curves).T # sample paths with means mu and covariance C
    else: Z = np.array([np.random.multivariate_normal(mu, k.K(X, X)) for i in range(n_curves) if not k.randomize()]).T

    df = pd.DataFrame(np.concatenate((X, Z), 1), columns = ['X'] + [n + 1 for n in range(Z.shape[1])])
    return ggplot(pd.melt(df, id_vars = ['X'], value_vars = list(range(Z.shape[1]))[1:])) +\
            geom_line(aes('X', 'value', color = 'variable'), show_legend = False)


def saveKernelSamples(ks, names, path, xlims = (-3., 3.), n_curves = 5, same_kernel = False):
    for k, n in zip(ks, names): sampleCurves(k, xlims, n_curves, same_kernel).savefig(f'{n}.png')


