import random
import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import pandas as pd
from focus import Focus

random.seed(42)
np.random.seed(42)


@pytest.fixture
def X():
    data = []

    for _ in range(300):
        # Generate random features
        features = [random.random() for _ in range(10)]
        data.append(features)

    # Create a pandas DataFrame from the data
    column_names = [f"feature_{i + 1}" for i in range(10)]
    df = pd.DataFrame(data, columns=column_names)
    return df


@pytest.fixture
def y():
    targets = []

    for _ in range(300):
        # Generate random features
        target = random.randint(0, 1)
        targets.append(target)

    # Create a pandas DataFrame from the data
    df = pd.DataFrame(targets)
    return df


@pytest.fixture
def dt_model(X, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X)
    return model


@pytest.fixture
def ab_model(X, y):
    model = AdaBoostClassifier(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


model_data = [
    (DecisionTreeClassifier(random_state=42).fit(X, y), X),
    (ab_model, X),
    (rf_model, X),
]


@pytest.mark.parametrize("model, X", model_data)
def test_prepare_features_by_perturb_direction(model, X):
    focus = Focus()
    direction = "positive"
    prepared_X = focus.prepare_features_by_perturb_direction(model, X, direction)
    assert len(prepared_X) < len(X)


def test_compute_gradient(model, X):
    focus = Focus()
    predictions = tf.constant([0, 1])
    to_optimize = [X]
    example_pred_class_index = tf.constant([[0, 0], [1, 1]])
    mask_vector = tf.constant([True, False])
    perturbed = tf.constant([[1, 2, 3], [7, 8, 9]])
    distance_weight = 0.01
    x_train = tf.constant([[1, 2, 3], [4, 5, 6]])
    distance_function = "euclidean"
    sigma = 10.0
    temperature = 1.0
    optimizer = tf.keras.optimizers.Adam()

    gradient = focus.compute_gradient(
        model,
        X,
        predictions,
        to_optimize,
        example_pred_class_index,
        mask_vector,
        perturbed,
        distance_weight,
        x_train,
        distance_function,
        sigma,
        temperature,
        optimizer,
    )

    assert isinstance(gradient, tf.Tensor)


def test_parse_class_tree(X):
    tree = MagicMock()
    X = np.array([[1, 2, 3], [4, 5, 6]])
    sigma = 0.5

    impurity_values = Focus.parse_class_tree(tree, X, sigma)

    assert isinstance(impurity_values, list)


def test_get_prob_classification_tree(X):
    tree = MagicMock()
    X = tf.constant([[1, 2, 3], [4, 5, 6]])
    sigma = 0.5

    prob_classification = Focus.get_prob_classification_tree(tree, X, sigma)

    assert isinstance(prob_classification, tf.Tensor)


def test_get_prob_classification_forest(model, X):
    focus = Focus()
    sigma = 0.5
    temperature = 1.0

    prob_classification = focus.get_prob_classification_forest(
        model, X, sigma, temperature
    )

    assert isinstance(prob_classification, tf.Tensor)


def test_filter_hinge_loss(model, X):
    focus = Focus()
    n_class = 2
    mask_vector = np.array([True, False])
    sigma = 0.5
    temperature = 1.0

    hinge_loss = focus.filter_hinge_loss(
        n_class,
        mask_vector,
        X,
        sigma,
        temperature,
        model,
    )

    assert isinstance(hinge_loss, tf.Tensor)
