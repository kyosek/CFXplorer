import pytest
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from focus import Focus

x_train, y_train = make_classification(
    n_samples=200, n_features=10, n_classes=2, random_state=42
)
x_test, y_test = make_classification(
    n_samples=100, n_features=10, n_classes=2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

ab_model = AdaBoostClassifier(random_state=42)
ab_model.fit(x_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

model_data = [
    (dt_model, x_test),
    (ab_model, x_test),
    (rf_model, x_test),
]

focus_model_data = [
    (dt_model, x_test, None, tf.keras.optimizers.Adam(), "euclidean", False),
    (ab_model, x_test, None, tf.keras.optimizers.Adam(), "l1", False),
    (rf_model, x_test, None, tf.keras.optimizers.Adam(), "euclidean", False),
    (dt_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", False),
    (ab_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", False),
    (rf_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", False),
    (dt_model, x_test, None, tf.keras.optimizers.Adam(), "cosine", True),
    (ab_model, x_test, None, tf.keras.optimizers.Adam(), "cosine", True),
    (rf_model, x_test, None, tf.keras.optimizers.Adam(), "l1", True),
    (dt_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", True),
    (ab_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", True),
    (rf_model, x_test, x_train, tf.keras.optimizers.Adam(), "mahalanobis", True),
    (dt_model, x_test, None, tf.keras.optimizers.Nadam(), "euclidean", False),
    (ab_model, x_test, None, tf.keras.optimizers.RMSprop(), "l1", False),
    (rf_model, x_test, None, tf.keras.optimizers.Ftrl(), "euclidean", False),
    (dt_model, x_test, x_train, tf.keras.optimizers.Ftrl(), "mahalanobis", False),
    (ab_model, x_test, x_train, tf.keras.optimizers.Nadam(), "mahalanobis", False),
    (rf_model, x_test, x_train, tf.keras.optimizers.SGD(), "mahalanobis", False),
    (dt_model, x_test, None, tf.keras.optimizers.SGD(), "cosine", True),
    (ab_model, x_test, None, tf.keras.optimizers.Adagrad(), "cosine", True),
    (rf_model, x_test, None, tf.keras.optimizers.Adadelta(), "l1", True),
    (dt_model, x_test, x_train, tf.keras.optimizers.Adamax(), "mahalanobis", True),
    (ab_model, x_test, x_train, tf.keras.optimizers.Adamax(), "mahalanobis", True),
    (rf_model, x_test, x_train, tf.keras.optimizers.Adadelta(), "mahalanobis", True),
]


@pytest.mark.parametrize("model, X", model_data)
def test_prepare_features_by_perturb_direction(model, X):
    focus = Focus()
    direction = "positive"
    prepared_x = focus.prepare_features_by_perturb_direction(model, X, direction)
    assert len(prepared_x) < len(X)


@pytest.mark.parametrize(
    "model, x_test, x_train, optimizer, distance_function, hyperparameter_tuning",
    focus_model_data,
)
def test_generate(
    model, x_test, x_train, optimizer, distance_function, hyperparameter_tuning
):
    """Test `generate` method by using multiple combinations of different parameters"""
    focus = Focus(
        num_iter=2,
        optimizer=optimizer,
        distance_function=distance_function,
        hyperparameter_tuning=hyperparameter_tuning,
    )
    if hyperparameter_tuning:
        best_perturb, unchanged_ever, best_distance = focus.generate(
            model, x_test, x_train
        )

        assert best_perturb.all() != x_test.all()
        assert isinstance(unchanged_ever, int)
        assert isinstance(best_distance, float)

    else:
        best_perturb = focus.generate(model, x_test, x_train)
        assert best_perturb.all() != x_test.all()
