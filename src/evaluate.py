from utils import safe_open
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


def generate_cf_stats(
    output_root: str,
    data_name: str,
    distance_function,
    unchanged_ever,
    counterfactual_examples,
    start_time,
):
    cf_stats = {
        "dataset": data_name,
        "distance_function": distance_function,
        "unchanged_ever": unchanged_ever,
        "mean_dist": np.mean(counterfactual_examples),
        "time (min)": np.round((end_time - start_time) / 60),
    }

    print("saving the text file")
    with safe_open(output_root + "_cf_stats.txt", "w") as gsout:
        json.dump(cf_stats, gsout)


def plot_perturbed(df_perturb: pd.DataFrame):
    f, ax = plt.subplots(figsize=(3, 7))
    sns.barplot(
        x=np.mean(df_perturb.iloc[:, 2:], axis=0), y=df_perturb.iloc[:, 2:].columns
    )
    ax.set(xlabel="Average perturbation per column")
