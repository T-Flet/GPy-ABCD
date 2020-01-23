import numpy as np
import pandas as pd
from plotnine import ggplot, geom_line, aes


def gg_plot(X, Y):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    if len(np.shape(Y)) == 1: Y = np.array(Y)[:, None]
    df = pd.DataFrame({'X': X[:, 0], 'Y': Y[:, 0]})
    return ggplot(df) + geom_line(aes('X', 'Y'), color='blue')
# print(gg_plot(range(20), [x**2 for x in range(20)]))

def gg_f_plot(f, X):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    Y = np.vectorize(f)(X)
    df = pd.DataFrame({'X': X[:, 0], 'Y': Y[:, 0]})
    return ggplot(df) + geom_line(aes('X', 'Y'), color='blue')
# print(gg_f_plot(lambda x: x**2, range(20)))


def cp_f_maker(f1, f2, cp, line_up = False):
    def cp_f(x):
        if x < cp: return f1(x)
        else: return f2(x)
    def cp_f_lined_up(x):
        if x < cp: return f1(x)
        else: return (f1(cp) - f2(cp)) + f2(x)
    return np.vectorize(cp_f_lined_up if line_up else cp_f)

def cw_f_maker(f1, f2, cw_start, cw_end, line_up = False):
    def cw_f(x):
        if x < cw_start or x > cw_end: return f1(x)
        else: return f2(x)
    def cw_f_lined_up(x):
        if x < cw_start: return f1(x)
        elif x > cw_end: return (cw_f_lined_up(cw_end) - f1(cw_end)) + f1(x)
        else: return (f1(cw_start) - f2(cw_start)) + f2(x)
    return np.vectorize(cw_f_lined_up if line_up else cw_f)


def generate_data(f, X, Y_weight = 1, noise_weight = 1):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    Y = np.vectorize(f)(X) * Y_weight + np.random.randn(*X.shape) * noise_weight
    return X, Y
# X, Y = generate_data(lambda x: x**2, range(20))
# print(gg_plot(X, Y))

def generate_changepoint_data(X, Y1_f, Y2_f, cp, Y_weight = 1, noise_weight = 1, line_up = False):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    Y = cp_f_maker(Y1_f, Y2_f, cp, line_up)(X) * Y_weight + np.random.randn(*X.shape) * noise_weight
    return X, Y
# X, Y = generate_changepoint_data(range(15), lambda x: 2*x, lambda x: x**2, 5, 1, 1, False)
# print(gg_plot(X, Y))

def generate_changewindow_data(X, Y1_f, Y2_f, cw_start, cw_end, Y_weight = 1, noise_weight = 1, line_up = False):
    if len(np.shape(X)) == 1: X = np.array(X)[:, None]
    Y = cw_f_maker(Y1_f, Y2_f, cw_start, cw_end, line_up)(X) * Y_weight + np.random.randn(*X.shape) * noise_weight
    return X, Y
# X, Y = generate_changewindow_data(range(15), lambda x: 2*x, lambda x: x**2, 5, 10, 1, 1, True)
# print(gg_plot(X, Y))
