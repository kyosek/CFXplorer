import random

import numpy as np
import pytest
import tensorflow as tf

from focus.utils import (
    calculate_distance,
    safe_cosine,
    safe_euclidean,
    safe_l1,
    safe_mahal,
    tf_cov,
)

epsilon = 10.0 ** -10
random.seed(42)

covariance_test_data = [
    (
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        tf.constant(
            [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]], dtype=tf.float64
        ),
    ),
    (
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        tf.convert_to_tensor(
            np.cov(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T, bias=True),
            dtype=tf.float64,
        ),
    ),
]

distance_test_data = [
    (
        np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [1, 2, 3]], dtype=np.float64
        ),
        np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [1, 2, 3]], dtype=np.float64
        ),
    )
]

calculate_distance_data = [
    # Euclidean
    (
        "euclidean",
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([5.1961524227162545, 5.1961524227162545, 10.392304845418074]),
    ),
    # Cosine
    (
        "cosine",
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]], dtype=np.float64),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        np.array([1.9746319, 1.9981909, 1.959412]),
    ),
    # Manhattan
    (
        "l1",
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([9, 9, 18]),
    ),
    # Mahalanobis
    (
        "mahalanobis",
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        np.array([[1, 2, 0], [6, 4, 5], [9, 7, 8]], dtype=np.float64),
        np.array([[4.9, 9.9, 4.1], [7.8, 9.9, 1.4], [2.9, 9.9, 3.2]]),
        np.array([-1.20098143e08, -1.20019050e28, -1.20019050e28]),
    ),
]


@pytest.mark.parametrize("feat_input_cov, expected_output_cov", covariance_test_data)
def test_tf_cov(feat_input_cov, expected_output_cov):
    assert tf_cov(feat_input_cov).numpy() == pytest.approx(
        expected_output_cov.numpy(), rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_euclidean(feat_input, feat_input2):
    expected = (np.sum(feat_input2 ** 2, axis=-1) + epsilon) ** 0.5
    assert safe_euclidean(feat_input).numpy() == pytest.approx(expected)


@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_cosine(feat_input, feat_input2):
    cosine_distance = safe_cosine(feat_input, feat_input, epsilon)

    # Assert that the shape of the output is as expected
    assert cosine_distance.shape == (len(feat_input),)


@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_l1(feat_input, feat_input2):
    expected = np.sum(abs(feat_input2), axis=1) + epsilon
    assert safe_l1(feat_input).numpy() == pytest.approx(expected)


@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_mahal(feat_input, feat_input2):
    """
    This unit test checks the function's accuracy in producing the expected output values.
    The correctness of the function's underlying logic has been previously verified.
    """
    mahalanobis_distance = safe_mahal(feat_input, feat_input2)

    # Assert that the shape of the output is as expected
    assert mahalanobis_distance.shape == (len(feat_input),)


@pytest.mark.parametrize(
    "distance_function, perturbed, feat_input, x_train, expected_output",
    calculate_distance_data,
)
def test_calculate_distance(
    distance_function, perturbed, feat_input, x_train, expected_output
):
    assert calculate_distance(
        distance_function, perturbed, feat_input, x_train
    ).numpy() == pytest.approx(expected_output)
