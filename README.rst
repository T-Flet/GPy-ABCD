GPy-ABCD
========

.. image:: https://img.shields.io/pypi/v/GPy-ABCD.svg
    :target: https://pypi.python.org/pypi/GPy-ABCD/
    :alt: Latest PyPI version

.. image:: https://pepy.tech/badge/gpy-abcd
    :target: https://pepy.tech/project/gpy-abcd
    :alt: Package Downloads

.. image:: https://img.shields.io/pypi/pyversions/GPy-ABCD.svg
    :target: https://pypi.python.org/pypi/GPy-ABCD/
    :alt: Python Versions

.. image:: https://github.com/T-Flet/GPy-ABCD/workflows/Python%20package/badge.svg
    :target: https://github.com/T-Flet/GPy-ABCD/actions?query=workflow%3A%22Python+package%22
    :alt: Build

.. image:: https://img.shields.io/pypi/l/GPy-ABCD.svg
    :target: https://github.com/T-Flet/GPy-ABCD/blob/master/LICENSE
    :alt: License

Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system

Briefly: a modelling system which consists in exploring a space of compositional kernels
(i.e. covariances of gaussian processes) built from a few carefully selected base ones,
returning the best fitting gaussian process models using them and generating simple text
interpretations of the fits based on the functional shapes of the final composed covariance
kernels and parameter values.

See the picture in `Usage` below to get a feeling for it and
read one of the papers on the original ABCD for details:

Lloyd, James Robert; Duvenaud, David Kristjanson; Grosse, Roger Baker; Tenenbaum, Joshua B.; Ghahramani, Zoubin (2014):
Automatic construction and natural-language description of nonparametric regression models.
In: National Conference on Artificial Intelligence, 7/27/2014, pp. 1242-1250.
Available online at https://academic.microsoft.com/paper/1950803081.


Installation
------------
::

    pip install GPy_ABCD

Usage
-----
The main function exported by this package is :code:`explore_model_space`;
note that if the :code:`parallel` argument is :code:`True` then the function should be
called from within a :code:`if __name__ == '__main__':`

::

    import numpy as np
    from GPy_ABCD import *

    if __name__ == '__main__':
        # Example data
        X = np.linspace(-10, 10, 101)[:, None]
        Y = np.cos( (X - 5) / 2 )**2 * X * 2 + np.random.randn(101, 1)

        # Main function call with suggested arguments
        best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y,
            start_kernels = standard_start_kernels, p_rules = production_rules_all,
            utility_function = BIC, restarts = 4, rounds = 2, buffer = 3, dynamic_buffer = True,
            verbose = True, parallel = True, optimiser = GPy_optimisers[0])


        # Typical output exploration printout

        for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

        print()

        # Explore the best 3 models in detail
        from matplotlib import pyplot as plt
        for bm in best_mods[:3]: model_printout(bm)

        # Perform some predictions
        predict_X = np.linspace(10, 15, 50)[:, None]
        preds = best_mods[0].predict(predict_X)
        print(preds)

        plt.show()


.. figure:: selected_output_example.png
    :align: center
    :figclass: align-center

    Selection of output from the above example

Importable elements from this package (refer to the section below for context):

- The :code:`GPModel` class
- The main function :code:`explore_model_space`
- The :code:`model_search_rounds` function to continue a search from where another left-off
- Single and list model fitting functions :code:`fit_one_model`, :code:`fit_model_list_not_parallel` and :code:`fit_model_list_parallel`
- The default start kernels :code:`standard_start_kernels` and production rules :code:`production_rules_all`, along with the same production rules grouped by type in a dictionary :code:`production_rules_by_type`
- The concrete :code:`KernelExpression` subclasses :code:`SumKE`, :code:`ProductKE` and :code:`ChangeKE`
- The frozensets of :code:`base_kerns` and :code:`base_sigmoids`

(The purpose of exporting elements in the last 3 lines is for users to create alternative sets of production
rules and starting kernel lists by mixing kernel expressions and raw strings of base kernels)

Project Structure
-----------------

Read the paper mentioned above for a full picture of what an ABCD system is, but, briefly,
it consists in exploring a space of compositional kernels built from a few carefully selected base ones,
returning the best fitting models using them and generating simple text interpretations of the fits based
on the functional shapes of the final composed covariance kernels and parameter values.

The key pillars of this project's ABCD system implementation structure are the following:

- :code:`Kernels.baseKernels` contains the "mathematical" base kernels (i.e. GPy kernel objects) for the whole machinery

    - This script also acts as a general configuration of what the system can use (including a few pre-packaged flags for certain behaviours)
    - Some of the base kernels are simply wrapped GPy-provided kernels (White-Noise, Constant and Squared-Exponential)
    - The others are either not present in GPy's default arsenal or are improved versions of ones which are (Linear which can identify polynomial roots and purely-Periodic standard-periodic kernel)
    - It contains sigmoidal kernels (both base sigmoids and indicator-like ones, i.e. sigmoidal hat/well) which are not used directly in the symbolic expressions but are substituted in by change-type kernels
    - It contains (multiple implementations of) change-point and change-window kernels which use the aforementioned sigmoidals
- :code:`KernelExpansion.kernelExpression` contains the "symbolic" kernel classes constituting the nodes with which to build complex kernel expressions in the form of trees

    - The non-abstract kernel expression classes are :code:`SumKE`, :code:`ProductKE` and :code:`ChangeKE`
    - :code:`SumKE` and :code:`ProductKE` are direct subclasses of the abstract class `SumOrProductKE` and only really differ in how they self-simplify and distribute over the other
    - :code:`ChangeKE` could be split into separate change-point and change-window classes, but a single argument difference allows full method overlap
    - :code:`SumOrProductKE` and :code:`ChangeKE` are direct subclasses of the abstract base class :code:`KernelExpression`
- The above kernel expression classes have a wide variety of methods providing the following general functionality in order to make the rest of the project light of ad-hoc functions:

    - They self-simplify when modified through the appropriate methods (they are symbolic expressions after all)
    - They can produce GPy kernel objects
    - They can line-up with and absorb fit model parameters from a matching GPy object
    - They can rearrange to a sum-of-products form
    - They can generate text interpretations of their sum-of-products form
- :code:`KernelExpansion.grammar` contains the various production rules and default kernel lists used in model space exploration
- :code:`Models.modelSearch` contains the system front-end elements:

    - The :code:`GPModel` class, which is where the GPy kernels/models interact with the symbolic kernel expressions
    - Functions to fit lists of models (the parallel version uses :code:`multiprocessing`'s :code:`Pool`, but alternative parallel frameworks' versions can be implemented here)
    - The :code:`explore_model_space` function, which is the point of it all
    - The :code:`model_search_rounds` function, which is used by the above but also meant to continue searching by building on past exploration results

Further Notes
-------------

- The important tests are in pytest scripts, but many other scripts are present and intended as functionality showcases or "tests by inspection"
- Additionally, pytest.ini has a two opposite configuration lines intended to be toggled to perform "real" tests vs other "by inspection" tests
- Please feel free to fork and expand this project since it is not the focus of my research and merely a component I need for part of it, therefore I will not be expanding its functionality in the near future

Possible expansion directions:

- Many "TODO" comments are present throughout the codebase
- Optimising ChangeWindow window-location fitting is an open issue (multiple implementations of change-window and the sigmoidal kernels they rely on have already been tried; see the commented-out declarations in baseKernels.py)
- The periodic kernel could be more stable in non-periodic-data fits (GPy's own as well)
- Making each project layer accept multidimensional data, starting from the GPy kernels (some already do)
- Expanding on the GPy side of things: add more methods to the kernels in order to make use of the full spectrum of GPy features (MCMC etc)
