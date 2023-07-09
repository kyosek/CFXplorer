import numpy as np
import tensorflow as tf


def safe_euclidean(matrix_diff, epsilon=10.0 ** -10) -> tf.Tensor:
    """
    Calculates the Euclidean distance between two matrices with a small epsilon added to prevent singularities.

    Args:
    matrix_diff: A tensor representing the difference between two matrices
    epsilon (float): A small number added to prevent singularities in the calculation (default 10.0 ** -10)

    Returns:
    tf.Tensor: A tensor representing the Euclidean distance between the two matrices
    """
    return (tf.reduce_sum(matrix_diff ** 2, axis=-1) + epsilon) ** 0.5


def safe_cosine(feat_input, perturbed, epsilon=10.0 ** -10) -> tf.Tensor:
    """
    Calculates cosine distance between two input arrays `feat_input` and `perturbed`
    while ensuring numerical stability with `epsilon`.

    Args:
        feat_input: The first input array to calculate cosine distance with.
        perturbed: The second input array to calculate cosine distance with.
        epsilon (float, optional): A small value added to the denominator to prevent division by zero.
        Defaults to 10 ** -10.

    Returns:
        tf.Tensor: The cosine distance between `feat_input` and `perturbed` as a tensor.
    """
    normalize_x1 = tf.nn.l2_normalize(feat_input)
    normalize_x2 = tf.nn.l2_normalize(perturbed)
    cosine_loss = tf.keras.losses.CosineSimilarity(
        axis=-1,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    dist = 1 - cosine_loss(normalize_x1, normalize_x2) + epsilon

    dist = tf.cast(tf.squeeze(dist), tf.float32)
    return dist


def safe_l1(matrix_diff, epsilon=10.0 ** -10) -> tf.Tensor:
    """
    Calculates the L1 (Manhattan) distance between two tensors with a small epsilon value
    added to prevent division by zero.

    Args:
    matrix_diff: Tensor to calculate L1 distance.
    epsilon: A small value to avoid division by zero (default: 10.0 ** -10).

    Returns:
    The L1 distance between the input tensors with the epsilon value added.
    """
    return tf.reduce_sum(tf.abs(matrix_diff), axis=1) + epsilon


def tf_cov(x_train) -> tf.Tensor:
    """
    Computes the covariance matrix of the input feature matrix x_train.

    Args:
    x_train: the train set

    Returns:
    cov_xx: a TensorFlow tensor of shape (num_features, num_features) representing the covariance matrix.
    """
    mean_x = tf.reduce_mean(x_train, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x_train), x_train) / tf.cast(
        tf.shape(x_train)[0], tf.float64
    )
    cov_xx = vx - mx
    return cov_xx


def safe_mahal(matrix_diff, x_train, epsilon=10.0 ** -10) -> tf.Tensor:
    """
    Calculates Mahalanobis distance using TensorFlow

    Args:
    matrix_diff: A tensor of shape (N, D) representing the difference between perturbed and feat_input
    x_train: The training data.
    epsilon: A scalar value to be added to the diagonal of covariance matrix
    to make it invertible.
    Returns:

    A tensor of shape (N,) representing the Mahalanobis distance between each sample
    and the mean of the training data.
    """
    covar = tf_cov(x_train)
    inv_covar = tf.linalg.inv(covar)
    return tf.reduce_sum(
        tf.multiply(tf.matmul(matrix_diff + epsilon, inv_covar), matrix_diff + epsilon),
        axis=1,
    )


def calculate_distance(
    distance_function: str,
    perturbed: tf.Variable,
    feat_input: np.ndarray,
    x_train: np.ndarray = None,
) -> tf.Tensor:
    """
    Calculates the distance between the perturbed and feat_input data using the specified distance function.

    Args:
    - distance_function (str): The distance function to use,
        can be either "euclidean", "cosine", "l1" or "mahalanobis".
    - perturbed (tf.Variable): The perturbed data.
    - feat_input (np.ndarray): The original feature data.
    - x_train (np.ndarray, optional): The training data, required for the Mahalanobis distance calculation.
    Defaults to None.

    Returns:
    - tf.Tensor: The calculated distance.

    Raises:
    - ValueError: If `distance_function` is "mahal" but `x_train` is not provided.
    """
    if distance_function == "euclidean":
        return safe_euclidean(perturbed - feat_input)
    elif distance_function == "cosine":
        return safe_cosine(feat_input, perturbed)
    elif distance_function == "l1":
        return safe_l1(perturbed - feat_input)
    elif distance_function == "mahalanobis":
        try:
            return safe_mahal(perturbed - feat_input, x_train)
        except ValueError:
            raise ValueError("x_train is empty")
