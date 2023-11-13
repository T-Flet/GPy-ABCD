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

*(Temporary note: the "failing" build badge above is due to the workflow pip not finding GPy 1.12.0 for some reason; the tests are successful)*

GPy-ABCD is a basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system.

Briefly, ABCD is a (Gaussian Process) modelling method which consists in exploring a space of modular kernels
(i.e. covariances) constructed by iteratively combining a small set of simple ones,
and returning the best fitting models using them;
due to its modularity, it is capable of generating simple text descriptions of the
fits based on the identified functional shapes.

The usefulness of ABCD is in identifying the underlying shape of data, but the process is
computationally expensive, therefore a typical use for it is in initial data overviews
(possibly on subsampled datasets for efficiency), then followed by more direct exploration
of its top few results' kernels (on the full dataset if subsampled before).

See the picture in `Usage` below for an example input/output and read the paper for further details:

`Fletcher, T., Bundy, A., & Nuamah, K. . GPy-ABCD: A Configurable Automatic Bayesian Covariance Discovery Implementation.
8th ICML Workshop on Automated Machine Learning (2021) <https://www.research.ed.ac.uk/en/publications/gpy-abcd-a-configurable-automatic-bayesian-covariance-discovery-i>`_



Installation
------------
::

    pip install GPy_ABCD



Usage
-----
The main function exported by this package is :code:`explore_model_space`;
see its description for parameter information and type hints.
Note that with the default :code:`model_list_fitter = fit_mods_parallel_processes` argument
the function should be called from within a :code:`if __name__ == '__main__':` for full OS-agnostic use.

A minimal example to showcase the various parameters follows:

.. code-block:: Python

    import numpy as np
    from GPy_ABCD import *


    if __name__ == '__main__':
        # Example data
        X = np.linspace(-10, 10, 101)[:, None]
        Y = np.cos((X - 5) / 2) ** 2 * X * 2 + np.random.randn(101, 1)

        # Main function call with default arguments
        best_mods, all_mods, all_exprs, expanded, not_expanded = explore_model_space(X, Y,
            start_kernels = start_kernels['Default'], p_rules = production_rules['Default'],
            utility_function = BIC, rounds = 2, beam = [3, 2, 1], restarts = 5,
            model_list_fitter = fit_mods_parallel_processes, optimiser = GPy_optimisers[0],
            verbose = True)

        print('\nFull lists of models by round:')
        for mod_depth in all_mods: print(', '.join([str(mod.kernel_expression) for mod in mod_depth]) + f'\n{len(mod_depth)}')

        print('\n\nTop-3 models\' details:')
        for bm in best_mods[:3]:
            model_printout(bm, plotly = False) # See the definition of this convenience function for examples of model details' extraction
            print('Prediction at X = 11:', bm.predict(np.array([11])[:, None]), '\n')

        from matplotlib import pyplot as plt
        plt.show() # Not required for plotly = True above



.. figure:: selected_output_example.png
    :align: center
    :figclass: align-center

    Selection of output from the above example

The directly importable elements from this package are essentially those required to customise any of the arguments of the
main model search function plus a few convenient tools (refer to the section below for context):

- The main function :code:`explore_model_space`
- The :code:`model_search_rounds` function to continue a search from where another left-off
- Functions to be used as  :code:`model_list_fitter` argument: :code:`fit_mods_not_parallel` and :code:`fit_mods_parallel_processes` (using :code:`multiprocessing`'s :code:`Pool`)
- The single-fit function :code:`fit_one_model`, on which the list-fitting functions above are built (so that a user may implement their preferred parallelisation)
- Non-searching single-fit functions: :code:`fit_kex` (which takes :code:`KernelExpression` inputs and simplifies them if required) and :code:`fit_GPy_kern` (which takes GPy kernel inputs and returns GPy GPRegression objects, not full GPy-ABCD GPModel ones)
- A few standard model-scoring functions: :code:`BIC`, :code:`AIC`, :code:`AICc`
- The aforementioned convenience function :code:`model_printout`
- The list of GPy 1.9.9 model fit optimisers :code:`GPy_optimisers`
- A few (named) choices of start kernels and production rules in the dictionaries :code:`start_kernels` and :code:`production_rules`
- The dictionary of available production rules grouped by type :code:`production_rules_by_type`
- The concrete :code:`KernelExpression` subclasses :code:`SumKE`, :code:`ProductKE` and :code:`ChangeKE`
- The frozensets of :code:`base_kerns` and :code:`base_sigmoids`

(The purpose of exporting elements in the last 3 lines is for users to create alternative sets of production
rules and starting kernels lists by mixing kernel expressions and raw strings of base kernels;
see the definitions of entries of the :code:`start_kernels` and :code:`production_rules` dictionaries for examples)



Project Structure
-----------------

A paper on GPy-ABCD and its differences from the original ABCD is planned, but for the time being read the paper mentioned above for a full picture of what an ABCD system is.

However, briefly, it consists in exploring a space of compositional kernels built from a few carefully selected base ones,
returning the best fitting models using them and generating simple text interpretations of the fits based
on the functional shapes of the final composed covariance kernels and parameter values.

The core components of this project's ABCD implementation are the following:

- :code:`Kernels.baseKernels` contains the "mathematical" base kernels (i.e. GPy kernel objects) for the whole machinery

  - Some of the base kernels are simply wrapped GPy-provided kernels (White-Noise, Constant and Squared-Exponential)
  - The others are either not present in GPy's default arsenal or are improved versions of ones which are (Linear which can identify polynomial roots and purely-Periodic standard-periodic kernel)
  - It contains sigmoidal kernels (both base sigmoids and indicator-like ones, i.e. sigmoidal hat/well) which are not used directly in the symbolic expressions but are substituted in by change-type kernels
  - It contains change-point and change-window kernels which use the aforementioned sigmoidals
- :code:`KernelExpression` contains the "symbolic" kernel classes constituting the nodes with which to build complex kernel expressions in the form of trees

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
- :code:`KernelExpansion.grammar` contains the various production rules and default starting kernel lists used in model space exploration
- :code:`Models.modelSearch` contains the system front-end elements:

  - The :code:`GPModel` class, which is where the GPy kernels/models interact with the symbolic kernel expressions
  - The aforementioned functions to fit lists of models :code:`fit_mods_not_parallel` and :code:`fit_mods_parallel_processes`
  - The :code:`explore_model_space` function, which is the point of it all
  - The :code:`model_search_rounds` function, which is used by the above but also meant to continue searching by building on past exploration results

Note: a :code:`config.py` file is present, and it contains a few global-behaviour-altering flags (e.g. enabling/disabling the Squared-Exponential kernel)



Further Notes
-------------

Generic:

- Please reach out if you have successfully used this project in your own research
- Feel free to fork and expand this project (pull requests are welcome) since it
  was only written because its particular features were required in a broader
  adaptive statistical modelling context,
  and therefore it is unlikely that its functionality will be expanded in the near future

Code-related:

- The important tests are in pytest scripts, but many other scripts are present and intended as functionality showcases or "tests by inspection"
- Additionally, pytest.ini has a two opposite configuration lines intended to be toggled to perform "real" tests vs other "by inspection" tests

Possible expansion directions:

- Many "TODO" comments are present throughout the codebase
- Optimising ChangeWindow window-location fitting is an open issue (multiple implementations of change-window and the sigmoidal kernels they rely on have already been tried; see the commented-out declarations in baseKernels.py inv ersions before v1.0)
- The periodic kernel could be more stable in non-periodic-data fits (GPy's own as well)
- Making each project layer accept multidimensional data, starting from the GPy kernels (some already do)
- Expanding on the GPy side of things: add more methods to the kernels in order to make use of the full spectrum of GPy features (MCMC etc)


