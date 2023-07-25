import time

from utils import (
    generate_example_data,
    plot_pca,
    prepare_plot_df,
    standardize_features,
    train_decision_tree_model,
)

from cfepy import Focus


def run_example():
    """
    Runs an example workflow to showcase the
    usage of different functions and modules.

    This function
        - generates example data
        - standardizes the features
        - trains a decision tree model
        - performs perturbation using the FOCUS algorithm
        - visualizes the results using PCA plots

    Returns:
        None: This function displays the plots
        but does not return any value.
    """
    start_time = time.time()

    X_train, X_test, y_train, y_test = generate_example_data(1000)
    X_train, X_test = standardize_features(X_train, X_test)
    model = train_decision_tree_model(X_train, y_train)

    focus = Focus(
        num_iter=1000,
        distance_function="cosine",
    )

    perturbed_feats = focus.generate(model, X_test, X_train)

    end_time = time.time()
    print("Finished!! ~{} min".format(round((end_time - start_time) / 60)))

    plot_df, focus_plot_df = prepare_plot_df(model, X_test, perturbed_feats)
    plot_pca(plot_df, focus_plot_df)


if __name__ == "__main__":
    run_example()
