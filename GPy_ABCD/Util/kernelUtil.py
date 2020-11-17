import numpy as np
import pandas as pd
from plotnine import ggplot, geom_line, aes


def sortOutTypePair(k1, k2):
    t1 = type(k1)
    t2 = type(k2)
    if t1 == t2: return {t1: [k1, k2]}
    else: return {t1: k1, t2: k2}


def sampleCurves(k):
    """
    Plot sample curves from a given kernel
    """
    X = np.linspace(-3., 3., 500)  # 500 points evenly spaced over [3,3]
    X = X[:, None]   # reshape X to make it n*D
    mu = np.zeros(500)  # vector of the means
    C = k.K(X, X)    # covariance matrix
    # Generate 20 sample path with mean mu and covariance C
    Z = np.random.multivariate_normal(mu, C, 20).T

    df = pd.DataFrame(np.concatenate((X, Z), 1), columns=['X'] + list(range(Z.shape[1])))
    print(ggplot(pd.melt(df, id_vars=['X'], value_vars=list(range(Z.shape[1]))[1:])) +
            geom_line(aes('X', 'value', color='variable')))


