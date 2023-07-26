CFXplorer
=========

**Deployment & Documentation & Stats & License**

.. image:: https://img.shields.io/pypi/v/focus-cfe.svg?color=brightgreen
   :target: https://pypi.org/project/focus-cfe/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/focus-cfe/badge/?version=latest
   :target: https://focus-cfe.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://pepy.tech/badge/focus-cfe
   :target: https://pepy.tech/project/focus-cfe
   :alt: Downloads

.. image:: https://codecov.io/gh/kyosek/CFXplorer/branch/master/graph/badge.svg?token=G5I7TJR0JQ
    :target: https://codecov.io/gh/kyosek/CFXplorer

.. image:: https://dl.circleci.com/status-badge/img/gh/kyosek/CFXplorer/tree/master.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/kyosek/CFXplorer/tree/master
    :alt: Circle CI

.. image:: https://api.codeclimate.com/v1/badges/93840d29606abb212051/maintainability
   :target: https://codeclimate.com/github/kyosek/focus-cfe/maintainability
   :alt: Maintainability

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/kyosek/CFXplorer
   :alt: pre-commit

.. image:: https://img.shields.io/github/license/kyosek/CFXplorer.svg
   :target: https://github.com/kyosek/CFXplorer/blob/master/LICENSE
   :alt: License

---------

CFXplorer generates optimal distance counterfactual explanations of the original data for the instances in tree‐based machine learning models.

This package is an implementation of `FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles (Lucic, et at. 2022) <https://arxiv.org/abs/1911.12199>`_.

Installation
------------

It is recommended to use **pip** or **conda** for installation. Please make sure
**the latest version** is installed:

.. code-block:: bash

   pip install CFXplorer            # normal install
   pip install --upgrade CFXplorer  # or update if needed

.. code-block:: bash

   conda install -c conda-forge CFXplorer

Requirements
------------

- Python 3.6+
- numpy>=1.19.1
- tensorflow>=2.0.0
- scikit-learn>=1.0.2

Usage
-----
**FOCUS counterfactual explanation generation with 3 Lines of Code**\ :

.. code-block:: python

    from cfxplorer import Focus
    # Initialize Focus instance with default values
    focus = Focus()

    # Generate counterfactual explanations for given tree model and features
    pertubed = focus.generate(tree_model, X)


Examples
--------

- Comprehensive examples can be found in the `examples folder <https://github.com/kyosek/CFXplorer/blob/master/examples/focus_example.py>`_.
- Kaggle notebook example can be found `here <https://www.kaggle.com/code/kyosukemorita/focus-example>`__.
- Below is demonstrated a comparison of before and after Focus is applied to feature set from the example given above.

.. image:: https://raw.githubusercontent.com/kyosek/focus/master/docs/plot.png
    :width: 800px
    :height: 400px
    :scale: 100 %
    :alt: Before and After FOCUS was applied to the features from above example.

Limitations of Focus class
--------------------------

- Currently, Focus class can only be applied to scikit-learn ``DecisionTreeClassifier``, ``RandomForestClassifier`` and ``AdaBoostClassifier``.
- While categorical features may be included in the feature set, it is important to note that the interpretation of changes in categorical features, such as transitioning from age 40 to 20, may not provide meaningful insights.
- The input features should be scaled to the range of 0 and 1 before applying Focus-cfe. Therefore, it is necessary to transform the features prior to using Focus. However, this scaling process may introduce some additional complexity when interpreting the features after applying Focus.

Documentation
-------------

The documentation can be found `here <https://cfxplorer.readthedocs.io/en/latest/>`__.

Contributing
------------

If you would like to contribute to the project, please refer to;

- `ISSUE_TEMPLATE <https://github.com/kyosek/CFXplorer/tree/master/.github/ISSUE_TEMPLATE>`_ for raising an issue
- `PULL_REQUEST_TEMPLATE.md <https://github.com/kyosek/CFXplorer/blob/master/.github/PULL_REQUEST_TEMPLATE.md>`_ for raising a PR

License
-------
This package is using the `Apache License 2.0 <https://github.com/kyosek/CFXplorer/blob/master/LICENSE>`_ license.
