import numpy as np
import pandas as pd
import pickle

import pytest

from src.counterfactual_explanation import (
    _parse_class_tree,
    get_prob_classification_tree,
    get_prob_classification_forest,
    filter_hinge_loss,
    compute_cfe,
)

dt_model = pickle.load(open("retrained_models/dt_cf_compas_num_data_train.pkl", "rb"))
rf_model = pickle.load(open("retrained_models/rf_cf_compas_num_data_train.pkl", "rb"))
ab_model = pickle.load(open("retrained_models/ab_cf_compas_num_data_train.pkl", "rb"))
compas_path = "data/cf_compas_num_data_test.tsv"
heloc_path = "data/cf_heloc_data_test.tsv"
shop_path = "data/cf_shop2_data_test.tsv"
wine_path = "data/cf_wine_data_test.tsv"

tree_data = [
    # COMPAS dataset
    (
        "dt",
        pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        1.0,
    ),
    # HELOC dataset
    (
        "dt",
        pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        1.0,
    ),
    # Shopping dataset
    (
        "dt",
        pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        10.0,
    ),
    # Wine dataset
    (
        "dt",
        pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        3.0,
    ),
]

forest_data = [
    # COMPAS dataset
    (
        "rf",
        pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        1.0,
        5.0,
    ),
    # HELOC dataset
    (
        "rf",
        pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        1.0,
        4.0,
    ),
    # Shopping dataset
    (
        "ab",
        pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        10.0,
        3.0,
    ),
    # Wine dataset
    (
        "ab",
        pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        3.0,
        1.0,
    ),
]


@pytest.mark.parametrize("model, feat_input, sigma", tree_data)
def test__parse_class_tree(model, feat_input, sigma):
    if model == "dt":
        leaf_nodes = _parse_class_tree(dt_model, feat_input, sigma)

    assert len(leaf_nodes) == 2
    assert len(leaf_nodes[0][0]) == len(feat_input)
    assert len(leaf_nodes[1][0]) == len(feat_input)


@pytest.mark.parametrize("model, feat_input, sigma", tree_data)
def test_get_prob_classification_tree(model, feat_input, sigma):
    if model == "dt":
        dt_prob_list = get_prob_classification_tree(dt_model, feat_input, sigma)

    assert dt_prob_list.shape == (len(feat_input), 2)


@pytest.mark.parametrize("model, feat_input, sigma, temperature", forest_data)
def test_get_prob_classification_forest(model, feat_input, sigma, temperature):
    if model == "rf":
        rf_softmax = get_prob_classification_forest(
            rf_model, feat_input, sigma, temperature
        )
        assert rf_softmax.shape == (len(feat_input), 2)

    elif model == "ab":
        ab_softmax = get_prob_classification_forest(
            ab_model, feat_input, sigma, temperature
        )
        assert ab_softmax.shape == (len(feat_input), 2)


@pytest.mark.parametrize("model, feat_input, sigma, temperature", forest_data)
def test_filter_hinge_loss(model, feat_input, sigma, temperature):
    indicator = np.zeros(len(feat_input))
    dt_hinge_loss = filter_hinge_loss(
        len(dt_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        dt_model,
    )
    rf_hinge_loss = filter_hinge_loss(
        len(rf_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        rf_model,
    )
    ab_hinge_loss = filter_hinge_loss(
        len(ab_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        ab_model,
    )

    assert dt_hinge_loss.shape == (len(feat_input), 2)
    assert rf_hinge_loss.shape == (len(feat_input), 2)
    assert ab_hinge_loss.shape == (len(feat_input), 2)


@pytest.mark.parametrize(
    "feat_input, decision_tree_model, random_forest_model, adaboost_model",
    [
        (
            pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[
                :, :-1
            ],
            pickle.load(open("retrained_models/dt_cf_compas_num_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/rf_cf_compas_num_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/ab_cf_compas_num_data_train.pkl", "rb")),
        ),
        (
            pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
            pickle.load(open("retrained_models/dt_cf_heloc_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/rf_cf_heloc_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/ab_cf_heloc_data_train.pkl", "rb")),
        ),
        (
            pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
            pickle.load(open("retrained_models/dt_cf_shop2_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/rf_cf_shop2_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/ab_cf_shop2_data_train.pkl", "rb")),
        ),
        (
            pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
            pickle.load(open("retrained_models/dt_cf_wine_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/rf_cf_wine_data_train.pkl", "rb")),
            pickle.load(open("retrained_models/ab_cf_wine_data_train.pkl", "rb")),
        ),
    ],
)
@pytest.mark.parametrize("distance_function", ["euclidean", "cosine", "l1", "mahal"])
def test_compute_cfe(
    feat_input,
    decision_tree_model,
    random_forest_model,
    adaboost_model,
    distance_function,
):
    opt = "adam"
    sigma = 1.0
    temperature = 5.0
    distance_weight = 0.01
    lr = 0.001
    num_itr = 2

    unchanged, cfe_distance, best_perturb = compute_cfe(decision_tree_model, feat_input, distance_function, opt, sigma,
                                                        temperature, distance_weight, num_itr, feat_input)

    assert isinstance(unchanged, int)
    assert isinstance(cfe_distance, np.ndarray)
    assert isinstance(best_perturb, np.ndarray)
    assert best_perturb.shape == feat_input.shape

    unchanged, cfe_distance, best_perturb = compute_cfe(random_forest_model, feat_input, distance_function, opt, sigma,
                                                        temperature, distance_weight, num_itr, feat_input)

    assert isinstance(unchanged, int)
    assert isinstance(cfe_distance, np.ndarray)
    assert isinstance(best_perturb, np.ndarray)
    assert best_perturb.shape == feat_input.shape

    unchanged, cfe_distance, best_perturb = compute_cfe(adaboost_model, feat_input, distance_function, opt, sigma,
                                                        temperature, distance_weight, num_itr, feat_input)

    assert isinstance(unchanged, int)
    assert isinstance(cfe_distance, np.ndarray)
    assert isinstance(best_perturb, np.ndarray)
    assert best_perturb.shape == feat_input.shape
