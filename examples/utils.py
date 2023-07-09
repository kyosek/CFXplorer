import random
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def set_random_seeds(seed_value):
    """
    Set random seeds for reproducibility.

    Args:
        seed_value (int): The seed value to set for random number generators.

    """
    random.seed(seed_value)
    np.random.seed(seed_value)


def generate_example_data(rows: int = 1000):
    """
    Generate random data with a binary target variable and 10 features.

    Args:
        rows (int): The number of rows in the generated dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the randomly generated data.

    """
    X, y = make_classification(n_samples=rows, n_features=10, n_classes=2, random_state=42)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def standardize_features(x_train, x_test):
    """
    Standardizes the features of the input data using Min-Max scaling.

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The training data.
        x_test (pandas.DataFrame or numpy.ndarray): The test data.

    Returns:
        tuple: A tuple containing two pandas DataFrames.
            - The first DataFrame contains the standardized features of the training data.
            - The second DataFrame contains the standardized features of the test data.
    """
    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform the data to perform feature scaling
    scaler = scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)
    scaled_x_test = scaler.transform(x_test)

    # Create a new DataFrame with standardized features
    standardized_train = pd.DataFrame(scaled_x_train)
    standardized_test = pd.DataFrame(scaled_x_test)

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


def prepare_plot_df(model, X, X_focus):
    """
    Prepares the data for plotting by performing PCA (Principal Component Analysis) on the input data.

    Args:
        model (object): A trained machine learning model capable of making predictions.
        X (array-like): The input data for which predictions are made.
        X_focus (array-like): Additional input data used for focus predictions.

    Returns:
        tuple: A tuple containing two pandas DataFrames.
            - The first DataFrame contains the PCA-transformed features of `X` and the corresponding predictions.
            - The second DataFrame contains the PCA-transformed features of `X_focus` and the corresponding focus predictions.
    """
    pca = PCA(n_components=2)

    predictions = pd.DataFrame(model.predict(X), columns=["predictions"])
    focus_predictions = pd.DataFrame(model.predict(X_focus), columns=["predictions"])

    pca.fit(X)
    pca_features = pd.DataFrame(pca.transform(X), columns=["pca1", "pca2"])
    pca_focus_features = pd.DataFrame(pca.transform(X_focus), columns=["pca1", "pca2"])

    return pd.concat([pca_features, predictions], axis=1), pd.concat(
        [pca_focus_features, focus_predictions], axis=1
    )


def plot_pca(plot_df, focus_plot_df):
    """
    Plots the PCA-transformed features and corresponding predictions before and after applying FOCUS.

    Args:
        plot_df (pandas.DataFrame): A DataFrame containing the PCA-transformed features and predictions before applying FOCUS.
        focus_plot_df (pandas.DataFrame): A DataFrame containing the PCA-transformed features and predictions after applying FOCUS.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.scatterplot(
        data=focus_plot_df, x="pca1", y="pca2", hue="predictions", ax=axes[0]
    )
    axes[0].set_title("After applying FOCUS")
    sns.scatterplot(data=plot_df, x="pca1", y="pca2", hue="predictions", ax=axes[1])
    axes[1].set_title("Before applying FOCUS")
    fig.suptitle("Prediction Before and After FOCUS comparison")
    plt.show()
