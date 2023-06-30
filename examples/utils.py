import random
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def set_random_seeds(seed_value):
    """
    Set random seeds for reproducibility.

    Args:
        seed_value (int): The seed value to set for random number generators.

    """
    random.seed(seed_value)
    np.random.seed(seed_value)


def generate_example_data(rows: int):
    """
    Generate random data with a binary target variable and 10 features.

    Args:
        rows (int): The number of rows in the generated dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the randomly generated data.

    """
    data = []

    set_random_seeds(42)

    for _ in range(rows):
        # Generate random features
        features = [random.uniform(0, 500) for _ in range(10)]

        # Generate random target variable (0 or 1)
        target = random.randint(0, 1)

        # Append the data row to the dataset
        data.append(features + [target])

    # Create a pandas DataFrame from the data
    column_names = [f"feature_{i + 1}" for i in range(10)] + ["target"]
    df = pd.DataFrame(data, columns=column_names)

    # Separate features and target variable
    X = df.drop(columns=["target"])
    y = df["target"].to_frame()

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)


def standardize_features(x_train, x_test):
    """
    Standardize features in the range of 0 and 1 using scikit-learn's MinMaxScaler.

    Args:
        data (pandas.DataFrame): The input data containing the features to be standardized.

    Returns:
        pandas.DataFrame: The standardized data with features in the range of 0 and 1.

    """
    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform the data to perform feature scaling
    scaler = scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)
    scaled_x_test = scaler.transform(x_test)

    # Create a new DataFrame with standardized features
    standardized_train = pd.DataFrame(scaled_x_train, columns=x_train.columns)
    standardized_test = pd.DataFrame(scaled_x_test, columns=x_test.columns)

    return standardized_train, standardized_test


def train_decision_tree_model(X_train, y_train):
    """
    Train a decision tree model using scikit-learn.

    Args:
        X_train (array-like or sparse matrix of shape (n_samples, n_features)): The training input samples.
        y_train (array-like of shape (n_samples,)): The target values for training.

    Returns:
        sklearn.tree.DecisionTreeClassifier: The trained decision tree model.

    """
    # Create and train the decision tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model
