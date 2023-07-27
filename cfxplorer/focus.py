"""
Generate counterfactual explanation for predictions of tree-based models.
"""
from __future__ import division, print_function

import os

import numpy as np
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from cfxplorer.utils import calculate_distance

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Focus:
    """
    FOCUS Lucic, et al. 2022 computes
    Counterfactual Explanations (CFE) using the gradient descent
    method for predictions of the tree-based models.

    Parameters
    ----------
    distance_function: str, optional (default="euclidean")
        Distance function - one of followings;
            - "euclidean"
            - "cosine"
            - "l1"
            - "mahalabobis"

    optimizer: Keras optimizer, optional (default=tf.keras.optimizers.Adam())
        Optimizer for gradient decent

    sigma: float, optional (default=10.0)
        Sigma hyperparameter value for hinge loss

    temperature: float, optional (default=1.0)
        Temperature hyperparameter value for hinge loss

    distance_weight: float, optional (default=0.01)
        Weight hyperparameter for distance loss

    lr: float, optional (default=0.001)
        Learning rate for gradient descent optimization

    num_iter: int, optional (default=100)
        Number of iterations for gradient descent optimization

    direction: str, optional (default="both")
        Direction of perturbation (e.g. both, positive and negative)

    hyperparameter_tuning: bool, optional (default=False)
        if True, generate method returns unchanged_ever and mean_distance

    verbose: int, optional (default=1)
        Verbosity mode.
            - 0: silent
            - else: print current number of iterations

    Reference
    ---------
    Lucic, A., Oosterhuis, H., Haned, H., & de Rijke, M. (2022, June).
    FOCUS: Flexible optimizable counterfactual explanations for tree ensembles.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 5, pp. 5313-5322).

    Examples
    --------
    - Initialize FOCUS on default parameters
    - Generate counterfactual explanations

    focus = Focus()

    cfe_features = focus.generate(model, X)
    """

    def __init__(
        self,
        distance_function="euclidean",
        optimizer=tf.keras.optimizers.Adam(),
        sigma=10.0,
        temperature=1.0,
        distance_weight=0.01,
        lr=0.001,
        num_iter=100,
        direction="both",
        hyperparameter_tuning=False,
        verbose=1,
    ):
        self.distance_function = distance_function
        self.optimizer = optimizer
        self.sigma = sigma
        self.temperature = temperature
        self.distance_weight = distance_weight
        self.lr = lr
        self.num_iter = num_iter
        self.direction = direction
        self.hyperparameter_tuning = hyperparameter_tuning
        self.verbose = verbose

    def generate(self, model, X, x_train=None):
        """
        Generate counterfactual explanations for the
        predictions from a tree-based model.

        Args:
        model: model object
            The machine learning model
                - DecisionTreeClassifier
                - RandomForestClassifier
                - AdaBoostClassifier

        X: numpy array
            The input feature to generate CFE

        x_train: numpy array, optional (default=None)
            The training data features
                - This will be used to calculate Mahalanobis distances

        Returns:
            The best perturbed features

        This method generates counterfactual explanations for the
        predictions made by a tree-based model.
        It uses the gradient descent method to optimize the input features
        based on a combination of hinge loss, approximate probability and a distance term.
        The `model` should be an instance of a tree-based model,
        such as DecisionTreeClassifier, RandomForestClassifier or AdaBoostClassifier.
        The `X` parameter represents the input features for which
        counterfactual explanations are desired.
        The `x_train` parameter is an optional argument that
        represents the training data features used
        to compute the approximate probability.

        The method returns the best perturbed features,
        which represent the optimized input features that
        result in counterfactual explanations.
        """
        X = Focus.prepare_features_by_perturb_direction(model, X, self.direction)

        perturbed = tf.Variable(
            initial_value=X,
            trainable=True,
            name="perturbed_features",
        )
        n_rows = len(X)
        distance_weight: np.ndarray = np.full(n_rows, self.distance_weight)
        to_optimize = [perturbed]
        mask_vector = np.ones(n_rows)
        best_perturb = np.zeros(perturbed.shape)
        best_distance = np.full(n_rows, np.inf)
        predictions = tf.constant(model.predict(X))

        for i in range(1, self.num_iter + 1):
            if self.verbose != 0:
                print(f"iteration {i}")

            grad = Focus.compute_gradient(
                model,
                X,
                predictions,
                to_optimize,
                mask_vector,
                perturbed,
                distance_weight,
                x_train,
                self.distance_function,
                self.sigma,
                self.temperature,
                self.optimizer,
            )

            self.optimizer.apply_gradients(
                zip(grad, to_optimize),
            )
            perturbed.assign(tf.clip_by_value(perturbed, 0, 1))

            distance = calculate_distance(self.distance_function, perturbed, X, x_train)

            cur_predicts = model.predict(perturbed.numpy())
            mask_vector = np.equal(predictions, cur_predicts).astype(np.float32)
            mask_flipped = predictions != cur_predicts

            distance_np = distance.numpy()
            mask_smaller_dist = distance_np < best_distance

            temp_dist = best_distance.copy()
            temp_dist[mask_flipped] = distance_np[mask_flipped]
            best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

            temp_perturb = best_perturb.copy()
            temp_perturb[mask_flipped] = perturbed[mask_flipped]
            best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

        if self.hyperparameter_tuning:
            return (
                best_perturb,
                len(best_distance[best_distance == np.inf]),
                np.mean(best_distance[best_distance != np.inf]),
            )

        else:
            print(
                f"The number of rows that are unchanged ever is "
                f"{len(best_distance[best_distance == np.inf])}"
            )
            print(
                f"The mean distance is "
                f"{np.mean(best_distance[best_distance != np.inf])}"
            )

            return best_perturb

    @staticmethod
    def prepare_features_by_perturb_direction(model, X: np.ndarray, direction: str):
        """
        Prepares the input data `X` based on the perturbation direction.

        Args:
            model (object): The model used for predicting the labels.
            X (np.ndarray): The input data to be prepared based on the perturbation direction.
            direction (str): The perturbation direction to consider.
                                Available options: "positive", "negative", "both".

        Returns:
            np.ndarray: The prepared input data based on the perturbation direction.

        Raises:
            ValueError: If an invalid `direction` is provided.

        This method filters and prepares the input data `X` based on the perturbation direction
        specified by the `direction` argument.
        It uses the provided `model` to predict the labels for the input data.
        The available options for `direction` are:
        - "positive":
            Returns the subset of input data where the model predicts the label as 0.
        - "negative":
            Returns the subset of input data where the model predicts the label as 1.
        - "both":
            Returns the input data as is without any filtering.

        Note that the `model` object should have a `predict` method
        that returns the predicted labels.
        """
        if direction == "positive":
            return X[model.predict(X) == 0]
        elif direction == "negative":
            return X[model.predict(X) == 1]
        elif direction == "both":
            return X
        else:
            raise ValueError(f"direction {direction} is not available")

    @staticmethod
    def compute_gradient(
        model,
        X,
        predictions,
        to_optimize,
        mask_vector,
        perturbed,
        distance_weight,
        x_train,
        distance_function,
        sigma,
        temperature,
        optimizer,
    ):
        """
        Computes the gradient of the loss function with respect to the variables to optimize.

        Returns:
        tf.Tensor: The computed gradient of the loss function with respect to the variables to optimize.

        This method computes the gradient of the loss function based on the provided inputs.
        It uses a TensorFlow GradientTape to record the operations for automatic differentiation.
        The loss function is defined as a combination of hinge loss, approximate probability, and a distance term.
        The gradient is then calculated with respect to the variables specified in the `to_optimize` list.
        """
        prediction_class_index = tf.stack(
            (tf.constant(np.arange(len(X), dtype=int)), predictions), axis=1
        )

        with tf.GradientTape(persistent=True) as tape:
            hinge_loss = Focus.filter_hinge_loss(
                len(model.classes_),
                mask_vector,
                perturbed,
                sigma,
                temperature,
                model,
            )
            approx_prob = tf.gather_nd(hinge_loss, prediction_class_index)
            distance = calculate_distance(distance_function, perturbed, X, x_train)
            hinge_approx_prob = tf.cast(mask_vector * approx_prob, tf.float32)
            loss = tf.reduce_mean(
                hinge_approx_prob + distance_weight * tf.cast(distance, tf.float32)
            )

        return tape.gradient(loss, to_optimize)

    @staticmethod
    def parse_class_tree(tree, X, sigma: float) -> list:
        """
        Compute impurity of each leaf node in a decision tree and approximate it using sigmoid function.

        Args:
        tree (DecisionTreeClassifier): Trained decision tree model.
        X (np.ndarray): Input feature values.
        sigma (float): Scaling factor to apply to sigmoid activation.

        Returns:
        list: A list of impurity values for each class label in the tree.
        """
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

                sigma = np.full(len(X), sigma)
                activation = tf.math.sigmoid((X[:, feature[i]] - threshold[i]) * sigma)

                left_node, right_node = 1.0 - activation, activation
                nodes[children_left[i]], nodes[children_right[i]] = (
                    cur_node * left_node,
                    cur_node * right_node,
                )

            else:
                max_class = np.argmax(values[i])
                leaf_nodes[max_class].append(cur_node)

        return leaf_nodes

    @staticmethod
    def get_prob_classification_tree(tree, X, sigma: float) -> tf.Tensor:
        """
        get_prob_classification_tree - computes the probability of each sample's classification in a decision tree

        Args:
        tree (DecisionTreeClassifier): a fitted decision tree model
        X (tf.Tensor): a tensor of input features
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

        leaf_nodes = Focus.parse_class_tree(tree, X, sigma)

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
                tf.reshape(X[0, :], shape=(1, -1))
            )  # can differ depending on particular samples used to train each tree

            correct_class = tf.constant(
                1, shape=(len(X)), dtype=tf.float32
            )  # prob(belong to correct class) = 100 since there's only one node
            incorrect_class = tf.constant(
                0, shape=(len(X)), dtype=tf.float32
            )  # prob(wrong class) = 0
            if only_class == 1.0:
                class_0 = incorrect_class
                class_1 = correct_class
            elif only_class == 0.0:
                class_0 = correct_class
                class_1 = incorrect_class
            else:
                raise ValueError("The class should be either 0 or 1")
            class_labels = [class_0, class_1]
            prob_stacked = tf.stack(class_labels, axis=1)
        return prob_stacked

    @staticmethod
    def get_prob_classification_forest(
        model, X: tf.Tensor, sigma: float, temperature: float
    ) -> tf.Tensor:
        """
        Calculate the softmax probabilities for classification for a random forest or AdaBoost model.

        Args:
        model: RandomForestClassifier or AdaBoostClassifier
            The trained decision tree model.
        X: tf.Tensor
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
            Focus.get_prob_classification_tree(estimator, X, sigma)
            for estimator in model.estimators_
        ][:100]

        if isinstance(model, AdaBoostClassifier):
            weights = model.estimator_weights_
        elif isinstance(model, RandomForestClassifier):
            weights = np.full(len(model.estimators_), 1 / len(model.estimators_))
        else:
            raise ValueError(
                "model object should be either AdaBoostClassifier or RandomForestClassifier"
            )

        logits = sum(weight * tree for weight, tree in zip(weights, dt_prob_list))

        temperature = np.full(len(X), temperature)
        if type(temperature) in [float, int]:
            expits = tf.exp(temperature * logits)
        else:
            expits = tf.exp(temperature[:, None] * logits)

        softmax = expits / tf.reduce_sum(expits, axis=1)[:, None]

        return softmax

    @staticmethod
    def filter_hinge_loss(
        n_class,
        mask_vector,
        X,
        sigma,
        temperature,
        model,
    ) -> tf.Tensor:
        """
        Calculates the filtered probabilities of each data point for the given model.

        Args:
        n_class (int): Number of classes.
        mask_vector (np.ndarray): A boolean mask indicating which data points should be considered.
        X (tf.Tensor): The feature input for the model.
        sigma (float): The value of sigma for computing the probabilities.
        temperature (float): The temperature to be used for the softmax function.
        model: The machine learning model;
            e.g., DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier.

        Returns:
        hinge_loss (tf.Tensor): The filtered probabilities of each data point.
        """
        n_input = X.shape[0]

        filtered_input = tf.boolean_mask(X, mask_vector)

        if not isinstance(model, DecisionTreeClassifier):
            filtered_loss = Focus.get_prob_classification_forest(
                model, filtered_input, sigma=sigma, temperature=temperature
            )
        elif isinstance(model, DecisionTreeClassifier):
            filtered_loss = Focus.get_prob_classification_tree(
                model, filtered_input, sigma
            )

        indices = np.where(mask_vector)[0]
        hinge_loss = tf.tensor_scatter_nd_add(
            np.zeros((n_input, n_class)),
            indices[:, None],
            filtered_loss,
        )
        return hinge_loss
