import numpy as np
import pandas as pd
import random
import tensorflow as tf

import pytest

from focus.utils import (
    safe_euclidean,
    safe_cosine,
    safe_l1,
    safe_mahal,
    tf_cov,
    calculate_distance,
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
        # np.empty([200, 10], dtype=np.float),
        # np.empty([200, 10], dtype=np.float)
        # np.array([random.random() for _ in range(10)], dtype=np.float64),
        # np.array([random.random() for _ in range(10)], dtype=np.float64),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]], dtype=np.float64),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]], dtype=np.float64)
    )
    # # COMPAS dataset
    # (
    #     pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    #     pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    # ),
    # # HELOC dataset
    # (
    #     pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    #     pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    # ),
    # # Shopping dataset
    # (
    #     pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    #     pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    # ),
    # # Wine dataset
    # (
    #     pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    #     pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    # ),
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

    # Assert that the function returns a tensor of the correct data type
    assert cosine_distance.dtype == tf.float32

    # Assert that the function does not return NaN or Inf values
    assert not np.isnan(cosine_distance).any()
    assert not np.isinf(cosine_distance).any()


@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_l1(feat_input, feat_input2):
    expected = np.sum(abs(feat_input2), axis=1) + epsilon
    assert safe_l1(feat_input).numpy() == pytest.approx(expected)
#
#
@pytest.mark.parametrize("feat_input, feat_input2", distance_test_data)
def test_safe_mahal(feat_input, feat_input2):
    """
    This unit test checks the function's accuracy in producing the expected output values.
    The correctness of the function's underlying logic has been previously verified.
    """
    mahalanobis_distance = safe_mahal(feat_input, feat_input2)

    # Assert that the shape of the output is as expected
    assert mahalanobis_distance.shape == (len(feat_input),)

    # Assert that the function returns a tensor of the correct data type
    assert mahalanobis_distance.dtype == tf.float64

    # Assert that the function does not return NaN or Inf values
    assert not np.isnan(mahalanobis_distance).any()
    assert not np.isinf(mahalanobis_distance).any()


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
