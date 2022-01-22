#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xtractree_demo_rf_dt_lightgbm.py

This file is a demo of the xtractree algorithm.
"""
__author__ = "Jeremy Charlier"
__contributor__ = "Renan Waroux"
__revised__ = "16 January 2022"

import click
import lightgbm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xtractree import XtracTree


def load_dataset():
    # Load data from scikit-learn breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Create train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    return X_train, X_test, y_train, y_test


def get_estimator(experiment):
    if experiment == 1:
        outfile = "output/dt_xtractree.py"

        df_params = {"max_depth": 10, "max_features": "auto", "random_state": 0}

        # Create estimator
        estimator = DecisionTreeClassifier(**df_params)

    if experiment == 2:
        outfile = "rf_xtractree.py"

        rdf_params = {
            "n_estimators": 10,
            "max_depth": None,
            "max_features": "auto",
            "n_jobs": -1,
            "random_state": 0,
        }

        # Create estimator
        estimator = RandomForestClassifier(rdf_params)
    if experiment == 3:
        outfile = "lgbm_xtractree.py"

        lgbm_params = {"max_depth": 3, "n_estimators": 3, "objective": "binary"}

        # Create estimator
        estimator = lightgbm.LGBMClassifier(lgbm_params)

    return outfile, estimator


@click.command()
@click.option(
    "--experiment",
    default=1,
    help="The expirement you want to run with the demo script",
)
def run_demo(experiment):
    outfile, estimator = get_estimator(experiment)
    X_train, X_test, y_train, y_test = load_dataset()

    # Fit the estimator on train dataset
    estimator.fit(X_train, y_train)

    print("\n --- TEST 1 ---")
    p = XtracTree(
        estimator, X_train, X_test, sample_id=6, sample_ids=[0, 1, 2], out=outfile
    )
    p.build_model()
    df_rules = p.sample_rules()

    print("\n --- TEST 2 ---")
    p = XtracTree(estimator, X_train, X_test, sample_id=6)
    df_rules = p.sample_rules(ndecisions=3)

    print("\n --- TEST 3 ---")
    p = XtracTree(estimator, X_train, X_test, sample_id=6)
    df = p.decisionsForForest(nDecisionsPerTree=5)

    return df_rules, df


if __name__ == "__main__":
    run_demo()
