import numpy as np
from focus import Focus
import time
import pickle
import pandas as pd
from utils import generate_example_data, train_decision_tree_model, standardize_features


def run_example():
    """
    Run example function:
        1. Load the data and model
        2. Generate Counterfactual Explanations (iterate num_iter times)
        3. Evaluate the generated counterfactual explanations and produce the txt file in results folder
    """

    start_time = time.time()

    X_train, X_test, y_train, y_test = generate_example_data(1000)
    X_train, X_test = standardize_features(X_train, X_test)
    model = train_decision_tree_model(X_train, y_train)

    focus = Focus(num_iter=10, distance_function="mahalanobis")

    best_perturb = focus.generate(model, X_test, X_train)

    end_time = time.time()
    print("Finished!! ~{} min".format(np.round((end_time - start_time) / 60)))


if __name__ == "__main__":
    run_example()
