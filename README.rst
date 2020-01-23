GPy-ABCD
========

.. image:: https://img.shields.io/pypi/v/GPy-ABCD.svg
    :target: https://pypi.python.org/pypi/GPy-ABCD
    :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/pyversions/GPy-ABCD.svg
   :target: https://pypi.python.org/pypi/GPy-ABCD/
   :alt: Python Versions

.. image:: https://img.shields.io/pypi/l/GPy-ABCD.svg
   :target: https://pypi.python.org/pypi/GPy-ABCD/
   :alt: License

Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system

(as in Lloyd, James Robert; Duvenaud, David Kristjanson; Grosse, Roger Baker; Tenenbaum, Joshua B.; Ghahramani, Zoubin (2014):
Automatic construction and natural-language description of nonparametric regression models.
In: National Conference on Artificial Intelligence, 7/27/2014, pp. 1242-1250.
Available online at https://academic.microsoft.com/paper/1950803081.)

Usage
-----
::

    import GPy_ABCD

    best_mods, all_mods, all_exprs = GPy_ABCD.find_best_model(X, Y,
        start_kernels = standard_start_kernels, p_rules = production_rules_all,
        restarts = 5, utility_function = 'BIC', rounds = 2, buffer = 4,
        verbose = False, parallel = True)


    # Typical output printout

    for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

    from matplotlib import pyplot as plt
    for bm in best_mods:
        print(bm.kernel_expression)
        print(bm.model.kern)
        print(bm.model.log_likelihood())
        print(bm.cached_utility_function)
        bm.model.plot()
        print(bm.interpret())

    predict_X = np.linspace(FROM, TO, BY)[:, None]
    preds = best_mods[0].predict(predict_X)
    print(preds)

    plt.show()

Note: if the :code:`parallel` argument is :code:`True` then the function should be
called from within a :code:`if __name__ == '__main__':`

Installation
------------
::

    pip install gpy_abcd

Requirements
^^^^^^^^^^^^

Python 3.7

See requirements.txt
