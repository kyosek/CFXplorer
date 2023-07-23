========================
Usage
========================

.. _installation:

Installation
------------

Installation using pip:

.. code-block:: console

   (.venv) $ pip install focus-cfe


Example
-------

.. code-block:: python

    from focus import Focus


    # Initialize Focus instance with default values
    focus = Focus()
    # Generate counterfactual explanations for given tree model and features
    pertubed = focus.generate(tree_model, X)
