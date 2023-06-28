import numpy as np
from focus import Focus
import time
import pickle
import pandas as pd
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("model_type", type=str)
parser.add_argument("num_iter", type=int, default=100)
parser.add_argument("sigma", type=float, default=1.0)
parser.add_argument("temperature", type=float, default=1.0)
parser.add_argument("distance_weight", type=float, default=0.01)
parser.add_argument("lr", type=float, default=0.001)
parser.add_argument(
    "opt", type=str, default="adam", help="Options are either adam or gd (as str)"
)
parser.add_argument("data_name", type=str)
parser.add_argument("distance_function", type=str)


def main(
    model_type: str,
    num_iter: int,
    sigma: float,
    temperature: float,
    distance_weight: float,
    lr: float,
    opt: str,
    data_name: str,
    distance_function: str,
):
    """
    Main function:
        1. Load the data and model
        2. Generate Counterfactual Explanations (iterate num_iter times)
        3. Evaluate the generated counterfactual explanations and produce the txt file in results folder
    """

    start_time = time.time()

    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    feat_input = feat_matrix[:, :-1]

    train_name = data_name.replace("test", "train")
    train_data = pd.read_csv("data/{}.tsv".format(train_name), sep="\t", index_col=0)
    x_train = np.array(train_data.iloc[:, :-1])

    model = pickle.load(
        open("retrained_models/" + model_type + "_" + train_name + ".pkl", "rb")
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    focus = Focus(num_iter=5)

    unchanged_ever, cfe_distance, best_perturb = focus.generate(model, feat_input)

    end_time = time.time()
    print("Finished!! ~{} min".format(np.round((end_time - start_time) / 60)))


if __name__ == "__main__":
    args = parser.parse_args()

    main(
        args.model_type,
        args.num_iter,
        args.sigma,
        args.temperature,
        args.distance_weight,
        args.lr,
        args.opt,
        args.data_name,
        args.distance_function,
    )
