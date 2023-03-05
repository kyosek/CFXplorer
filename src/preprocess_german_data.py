import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split


def data_modification():
    """
    Downloaded the data (german.data) from https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data).
    This function modifies it in order to be used for the experiments.
    The modification follows Lucic et al. 2022, but still keep the categorical features with one hot encoding.
    Procedure:
        1. load the data
        2. Rename columns
        3. Make the target variable 0 and 1 instead of 1 (good) and 2 (bad)
        4. Split train and test data
        5. Normalise all the features
        6. Save the data
    """
    ohe = OneHotEncoder(handle_unknown="ignore", drop="first")

    cat_cols = [
        "existing_checking",
        "credit_history",
        "purpose",
        "savings",
        "employment_since",
        "status_gender",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
        "job",
        "telephone",
        "foreign_worker",
    ]

    # 1. load the data
    df = pd.read_csv("data/german.data", sep=" ")

    # 2. rename columns
    df.columns = [
        "existing_checking",
        "month_duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment_since",
        "installment_rate",
        "status_gender",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "liable_people_count",
        "telephone",
        "foreign_worker",
        "target",
    ]

    # 3. Make the target variable 0 and 1 instead of 1 (good) and -1 (bad)
    df["target"] = np.where(df["target"] == 1, -1, 1)

    # 4. Split train and test data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    ohe_train_df = pd.DataFrame(
        ohe.fit_transform(train_df[cat_cols]).toarray(),
        columns=ohe.get_feature_names_out(),
    )
    ohe_test_df = pd.DataFrame(
        ohe.transform(test_df[cat_cols]).toarray(), columns=ohe.get_feature_names_out()
    )

    train_df, test_df = train_df.drop(labels=cat_cols, axis=1), test_df.drop(
        labels=cat_cols, axis=1
    )

    # 5. Normalise all the features
    normalised_train_df = pd.DataFrame(
        Normalizer().fit_transform(X=train_df.drop(labels=["target"], axis=1).values)
    )
    normalised_train_df.columns = train_df.drop(labels=["target"], axis=1).columns
    normalised_train_df = pd.concat([normalised_train_df, ohe_train_df], axis=1)
    normalised_train_df["target"] = train_df["target"]

    normalised_test_df = pd.DataFrame(
        Normalizer().transform(X=test_df.drop(labels=["target"], axis=1).values)
    )
    normalised_test_df.columns = test_df.drop(labels=["target"], axis=1).columns
    normalised_test_df = pd.concat([normalised_test_df, ohe_test_df], axis=1)
    normalised_test_df["target"] = test_df["target"]

    print("train df distribution")
    print(normalised_train_df.describe())
    print("test df distribution")
    print(normalised_test_df.describe())

    # 6. Save the data
    normalised_train_df.to_csv("data/cf_german_train.tsv", sep="\t")
    normalised_test_df.to_csv("data/cf_german_test.tsv", sep="\t")


if __name__ == "__main__":
    data_modification()
