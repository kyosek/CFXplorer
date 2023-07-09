FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles
==========================================================================

**Deployment & Documentation & Stats & License**

.. image:: https://img.shields.io/pypi/v/pyod.svg?color=brightgreen
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version

-----

This library is an implementation of [`FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles`](https://arxiv.org/abs/1911.12199).

TLDR; FOCUS generates optimal distance counterfactual explanations to the original data for all the instances in tree‐based machine learning models.

**FOCUS counterfactual explanation generation with 5 Lines of Code**\ :

.. code-block:: python
from focus import Focus

# Initialize Focus instance
focus = Focus()
# Generate counterfactual explanation
pertubed = focus.generate(tree_model, X)

Installation
^^^^^^^^^^^^

It is recommended to use **pip** or **conda** for installation. Please make sure
**the latest version** is installed:

.. code-block:: bash

   pip install focus            # normal install
   pip install --upgrade focus  # or update if needed

.. code-block:: bash

   conda install -c conda-forge focus
