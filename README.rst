GPy-ABCD
========

.. image:: https://img.shields.io/pypi/v/GPy-ABCD.svg
    :target: https://pypi.python.org/pypi/GPy-ABCD/
    :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/pyversions/GPy-ABCD.svg
    :target: https://pypi.python.org/pypi/GPy-ABCD/
    :alt: Python Versions

.. image:: https://img.shields.io/pypi/l/GPy-ABCD.svg
    :target: https://github.com/T-Flet/GPy-ABCD/blob/master/LICENSE
    :alt: License

.. image:: https://github.com/T-Flet/GPy-ABCD/workflows/Python%20package/badge.svg
    :target: https://github.com/T-Flet/GPy-ABCD/actions?query=workflow%3A%22Python+package%22
    :alt: Build

Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system

(as in Lloyd, James Robert; Duvenaud, David Kristjanson; Grosse, Roger Baker; Tenenbaum, Joshua B.; Ghahramani, Zoubin (2014):
Automatic construction and natural-language description of nonparametric regression models.
In: National Conference on Artificial Intelligence, 7/27/2014, pp. 1242-1250.
Available online at https://academic.microsoft.com/paper/1950803081.)

Usage
-----
::

    import numpy as np
    from GPy_ABCD import *

    if __name__ == '__main__':
        X = np.linspace(-10, 10, 101)[:, None]
        Y = np.cos( (X - 5) / 2 )**2 * X * 2 + np.random.randn(101, 1)

        best_mods, all_mods, all_exprs = find_best_model(X, Y,
            start_kernels = standard_start_kernels, p_rules = production_rules_all,
            restarts = 5, utility_function = 'BIC', rounds = 2, buffer = 3,
            dynamic_buffer = True, verbose = False, parallel = True)

        # Typical full output printout

        for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

        from matplotlib import pyplot as plt
        for bm in best_mods[:3]:
            print(bm.kernel_expression)
            print(bm.model.kern)
            print(bm.model.log_likelihood())
            print(bm.cached_utility_function)
            bm.model.plot()
            print(bm.interpret())

        predict_X = np.linspace(10, 15, 50)[:, None]
        preds = best_mods[0].predict(predict_X)
        print(preds)

        plt.show()

Note: if the :code:`parallel` argument is :code:`True` then the function should be
called from within a :code:`if __name__ == '__main__':`

Installation
------------
::

    pip install GPy_ABCD

Requirements
^^^^^^^^^^^^

Python 3.7

See requirements.txt
