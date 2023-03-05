import tensorflow as tf
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import calculate_distance


def _parse_class_tree(tree, feat_input: np.ndarray, sigma: float) -> list:
    """
    Compute impurity of each leaf node in a decision tree and approximate it using sigmoid function.

    Parameters:
    tree (DecisionTreeClassifier): Trained decision tree model.
    feat_input (np.ndarray): Input feature values.
    sigma (float): Scaling factor to apply to sigmoid activation.

    Returns:
    list: A list of impurity values for each class label in the tree.
    """
    # Code is adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(len(tree.classes_))]

    node_depth = np.zeros(shape=n_nodes, dtype=np.int32)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        cur_node = nodes[i]
        if children_left[i] != children_right[i]:

            if cur_node is None:
                cur_node = 1.0

            sigma = np.full(len(feat_input), sigma)
            activation = tf.math.sigmoid(
                (feat_input[:, feature[i]] - threshold[i]) * sigma
            )

            left_node, right_node = 1.0 - activation, activation
            nodes[children_left[i]], nodes[children_right[i]] = (
                cur_node * left_node,
                cur_node * right_node,
            )

        else:
            max_class = np.argmax(values[i])
            leaf_nodes[max_class].append(cur_node)

    return leaf_nodes


def get_prob_classification_tree(tree, feat_input, sigma: float) -> tf.Tensor:
    """
    get_prob_classification_tree - computes the probability of each sample's classification in a decision tree

    Args:
    tree (DecisionTreeClassifier): a fitted decision tree model
    feat_input (tf.Tensor): a tensor of input features
    sigma (float): parameter for Gaussian smoothing
    Outputs:

    Returns:
    prob_stacked (tf.Tensor): a tensor of probabilities for each sample's classification
    This function returns the probabilities of each sample's classification in a decision tree.
    It calculates the impurities of the leaf nodes that each sample falls into, and then
    computes the sum of these impurities for each class. If the tree has only one node,
    the probability of the correct class is set to 1 and the probability of the incorrect class is set to 0.
    The final result is a tensor of stacked probabilities for each sample's classification.
    """

    leaf_nodes = _parse_class_tree(tree, feat_input, sigma)

    if tree.tree_.node_count > 1:
        prob_list = [sum(leaf_nodes[c_i]) for c_i in range(len(tree.classes_))]
        i = 0
        while i < len(prob_list):
            if prob_list[i].numpy().all() == 0:
                prob_list.pop(i)
            else:
                i += 1

        prob_stacked = tf.stack(prob_list, axis=-1)

    else:  # sometimes tree only has one node
        only_class = tree.predict(
            tf.reshape(feat_input[0, :], shape=(1, -1))
        )  # can differ depending on particular samples used to train each tree

        correct_class = tf.constant(
            1, shape=(len(feat_input)), dtype=tf.float32
        )  # prob(belong to correct class) = 100 since there's only one node
        incorrect_class = tf.constant(
            0, shape=(len(feat_input)), dtype=tf.float32
        )  # prob(wrong class) = 0
        if only_class == 1.0:
            class_0 = incorrect_class
            class_1 = correct_class
        elif only_class == 0.0:
            class_0 = correct_class
            class_1 = incorrect_class
        else:
            raise ValueError
        class_labels = [class_0, class_1]
        prob_stacked = tf.stack(class_labels, axis=1)
    return prob_stacked


def get_prob_classification_forest(
    model, feat_input: tf.Tensor, sigma: float, temperature: float
) -> tf.Tensor:
    """
    Calculate the softmax probabilities for classification for a random forest or AdaBoost model.

    Args:
    model: RandomForestClassifier or AdaBoostClassifier
        The trained decision tree model.
    feat_input: tf.Tensor
        The input feature matrix.
    sigma: float
        The sigma value used to compute the activation of nodes in each decision tree.
    temperature: float
        The temperature to adjust the scale of the logits.

    Returns:
    tf.Tensor
        The softmax probabilities of the classification.
    """
    dt_prob_list = [
        get_prob_classification_tree(estimator, feat_input, sigma)
        for estimator in model.estimators_
    ][:100]

    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_
    elif isinstance(model, RandomForestClassifier):
        weights = np.full(len(model.estimators_), 1 / len(model.estimators_))

    logits = sum(weight * tree for weight, tree in zip(weights, dt_prob_list))

    temperature = np.full(len(feat_input), temperature)
    if type(temperature) in [float, int]:
        expits = tf.exp(temperature * logits)
    else:
        expits = tf.exp(temperature[:, None] * logits)

    softmax = expits / tf.reduce_sum(expits, axis=1)[:, None]

    return softmax


def filter_hinge_loss(
    n_class, mask_vector, feat_input, sigma, temperature, model
) -> tf.Tensor:
    """
    Calculates the filtered probabilities of each data point for the given model.

    Args:
    - n_class (int): Number of classes.
    - mask_vector (np.ndarray): A boolean mask indicating which data points should be considered.
    - feat_input (tf.Tensor): The feature input for the model.
    - sigma (float): The value of sigma for computing the probabilities.
    - temperature (float): The temperature to be used for the softmax function.
    - model: The machine learning model (e.g., DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier).

    Returns:
    - hinge_loss (tf.Tensor): The filtered probabilities of each data point.
    """
    n_input = feat_input.shape[0]

    filtered_input = tf.boolean_mask(feat_input, mask_vector)

    if not isinstance(model, DecisionTreeClassifier):
        filtered_loss = get_prob_classification_forest(
            model, filtered_input, sigma=sigma, temperature=temperature
        )
    elif isinstance(model, DecisionTreeClassifier):
        filtered_loss = get_prob_classification_tree(model, filtered_input, sigma)

    indices = np.where(mask_vector)[0]
    hinge_loss = tf.tensor_scatter_nd_add(
        np.zeros((n_input, n_class)),
        indices[:, None],
        filtered_loss,
    )
    return hinge_loss


def compute_cfe(
    model,
    feat_input,
    distance_function: str,
    optimizer,
    sigma_val: float,
    temperature_val: float,
    distance_weight_val: float,
    num_iter=100,
    x_train=None,
    verbose=1,
):
    """
    This function computes Counterfactual Explanations (CFE) using the gradient descent method.

    Args:
    model: The machine learning model (e.g., DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier).
    feat_input: numpy array, the input feature to generate CFE
    distance_function: str, distance function - one of "euclidean", "cosine", "l1" and "mahal"
    optimizer: keras optimizer
    sigma_val: float, sigma value for hinge loss
    temperature_val: float, temperature value for hinge loss
    distance_weight_val: float, weight value for distance loss
    lr: float, learning rate for gradient descent optimization
    num_iter: int, number of iterations for gradient descent optimization (default=100)
    x_train: numpy array, the training data used to fit the original model
    verbose: int, verbosity of the function (default=1)

    Returns:
    tuple, number of examples that remain unchanged, the cfe distances for the changed examples and the best perturb
    """
    perturbed = tf.Variable(
        initial_value=feat_input,
        trainable=True,
        name="perturbed_features",
    )
    n_examples = len(feat_input)
    distance_weight: np.ndarray = np.full(n_examples, distance_weight_val)
    to_optimize = [perturbed]
    mask_vector = np.ones(n_examples)
    best_perturb = np.zeros(perturbed.shape)
    best_distance = np.full(n_examples, np.inf)
    perturb_iteration_found = np.full(n_examples, 1000 * num_iter, dtype=int)
    predictions = tf.constant(model.predict(feat_input), dtype=tf.int64)
    example_index = tf.constant(np.arange(n_examples, dtype=int))
    example_pred_class_index = tf.stack((example_index, predictions), axis=1)

    with tf.GradientTape(persistent=True) as tape:
        for i in range(num_iter):
            if verbose != 0:
                print(f"iteration {i}")

            hinge_loss = filter_hinge_loss(
                len(model.classes_),
                mask_vector,
                perturbed,
                sigma_val,
                temperature_val,
                model,
            )
            approx_prob = tf.gather_nd(hinge_loss, example_pred_class_index)
            distance = calculate_distance(
                distance_function, perturbed, feat_input, x_train
            )
            hinge_approx_prob = tf.cast(mask_vector * approx_prob, tf.float32)
            loss = tf.reduce_mean(
                hinge_approx_prob + distance_weight * tf.cast(distance, tf.float32)
            )

            grad = tape.gradient(loss, to_optimize)

            optimizer.apply_gradients(
                zip(grad, to_optimize),
            )
            perturbed.assign(tf.clip_by_value(perturbed, 0, 1))

            distance = calculate_distance(
                distance_function, perturbed, feat_input, x_train
            )

            cur_predict = model.predict(perturbed.numpy())
            mask_vector = np.equal(predictions, cur_predict).astype(np.float32)
            idx_flipped = np.flatnonzero(mask_vector == 0)
            mask_flipped = (predictions != cur_predict)

            perturb_iteration_found[idx_flipped] = np.minimum(
                i, perturb_iteration_found[idx_flipped]
            )

            distance_np = distance.numpy()
            mask_smaller_dist = (distance_np < best_distance)

            temp_dist = best_distance.copy()
            temp_dist[mask_flipped] = distance_np[mask_flipped]
            best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

            temp_perturb = best_perturb.copy()
            temp_perturb[mask_flipped] = perturbed[mask_flipped]
            best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

            unchanged_ever = len(best_distance[best_distance == np.inf])
            cfe_distance = np.mean(best_distance[best_distance != np.inf])

        return unchanged_ever, cfe_distance, best_perturb
