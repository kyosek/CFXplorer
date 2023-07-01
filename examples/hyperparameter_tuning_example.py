import optuna
import tensorflow as tf
from focus import Focus
from utils import (
    generate_example_data,
    train_decision_tree_model,
    standardize_features,
)


def objective(trial):
    """
    This function is an objective function for hyperparameter tuning using optuna.
    It explores the hyperparameter sets and evaluates the result on a given model and dataset
    Mean distance and number of unchanged instances are used for the evaluation.

    Args:
    trial (optuna.Trial): Object that contains information about the current trial, including hyperparameters.

    Returns:
    Mean CFE distance + number of unchanged instances squared -
    This is the objective function for hyperparameter optimization

    * Note: typically we want to minimise a number of unchanged first, so penalising the score by having squared number.
    Also, to not distort this objective, having the mean distance divided by 100.
    """
    X_train, X_test, y_train, y_test = generate_example_data(1000)
    X_train, X_test = standardize_features(X_train, X_test)
    model = train_decision_tree_model(X_train, y_train)

    focus = Focus(
        num_iter=1000,
        distance_function="euclidean",
        sigma=trial.suggest_int("sigma", 1, 20, step=1.0),
        temperature=0,  # DT models do not use temperature
        distance_weight=round(
            trial.suggest_float("distance_weight", 0.01, 0.1, step=0.01), 2
        ),
        lr=round(trial.suggest_float("lr", 0.001, 0.01, step=0.001), 3),
        optimizer=tf.keras.optimizers.RMSprop(),
        hyperparameter_tuning=True,
        verbose=0
    )

    best_perturb, unchanged_ever, cfe_distance = focus.generate(model, X_test)

    print(f"Unchanged: {unchanged_ever}")
    print(f"Mean distance: {cfe_distance}")

    return cfe_distance / 100 + pow(unchanged_ever, 2)


if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial

    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
