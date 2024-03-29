========================
Usage
========================

.. _installation:

Installation
------------

Installation using pip:

.. code-block:: console

   (.venv) $ pip install CFXplorer


Example
-------
``Focus`` can generates perturbed feature set ``X'`` given tree-based model - ``tree_model`` and feature set - ``X``.

.. code-block:: python

    from cfxplorer import Focus


    # Initialize Focus instance with default values
    focus = Focus()

    # Generate counterfactual explanations for given tree model and features
    pertubed = focus.generate(tree_model, X)
