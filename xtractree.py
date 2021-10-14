#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""xtractree.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_curve,
                             auc)
from sklearn.utils import Bunch
import pickle as pkl

# pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)


class XtracTree:
    def __init__(self, estimator, x_train, x_test,
                 sample_id=None, sample_ids=None, out=None):
        self.estimator = estimator
        self.x_train = x_train
        self.x_test = x_test
        self.sample_id = sample_id
        self.sample_ids = sample_ids
        self.out = out


    def build_dtree_rules(estimator, n_trees, x_train, l2w):
        """ Extract decision rules with splitting thresholds and probabilities.
        """
        # we can get useful information about the tree structure
        # using estimator.tree.__getstate__()['nodes']
        # or import sklearn, help(sklearn.tree._tree.Tree)
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        n_node_samples = estimator.tree_.n_node_samples
        value = estimator.tree_.value
        weighted_n_node_samples = estimator.tree_.weighted_n_node_samples

        node_depth = np.zeros(shape = n_nodes, dtype = np.int64)
        is_leaves = np.zeros(shape = n_nodes, dtype = bool)
        stack = [(0, -1)] # seed is the root node id and its parent depth

        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # if we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        for i in range(n_nodes):
            if is_leaves[i]:
                irow = "        if state == %s: return %s\n" % (
                    i, ((value[i][0,1] / weighted_n_node_samples[i]) / n_trees)
                    )
            else:
                irow = "        if state == %s: state = (%s if x['%s'] <= %s else %s)\n" % (
                    i,
                    children_left[i],
                    x_train.columns[feature[i]],
                    threshold[i],
                    children_right[i]
                )
            l2w.append(irow)
        return l2w


    def build_model(self):
        """Build the global architecture of the bagging algorithm.
        It is designed such that it is an executable if self.out is 
        saved as .py file. 
        """
        # we write the first line of the file
        # we store the content of the file to be written in l2w
        l2w = ["import numpy as np\n"]
        l2w.append("\n")
        l2w.append("def estimator_tree(x, num_tree):\n")
        l2w.append("    if num_tree == %s:\n" % 0)
        l2w.append("        state = %s\n" % 0)

        if 'RandomForestClassifier' in str(type(self.estimator)):
            n_trees = len(self.estimator.estimators_)
            for n in range(n_trees):
                if n == 0:
                    l2w = XtracTree.build_dtree_rules(
                        self.estimator.estimators_[n],
                        n_trees, self.x_train, l2w)
                else:
                    l2w.append("    elif num_tree == %s:\n" % n)
                    l2w.append("        state = %s\n" % 0)
                    l2w = XtracTree.build_dtree_rules(
                        self.estimator.estimators_[n],
                        n_trees, self.x_train, l2w)
        else:
            # the estimator is a decision tree
            # we can pass it directly to build_tree_rules
            n_trees = 1
            l2w = XtracTree.build_dtree_rules(
                        self.estimator,
                        n_trees, self.x_train, l2w)

        # we write at the bottom the predict rule for the fixed model
        l2w.append("\n\ndef estimator_predict(x):\n")

        # we initialize the proba values at 0
        l2w.append("    predict = 0.0\n")
        l2w.append("    for i in range(%s):\n" % n_trees)
        l2w.append("        predict += estimator_tree(x, i)\n")
        l2w.append("    return predict\n")

        if self.out is not None:
            with open(self.out, 'w') as the_file:
                the_file.write("".join(l2w))
                the_file.close()
        else:
            print(l2w)

        return None


    def display_rule_per_estimator(estimator, X_test, sample_id, l2r):
        """Display the decision path per tree contained in the estimator for 1 sample.
        """
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        node_indicator = estimator.decision_path(X_test)

        # we have the leaves ids reached by each sample.
        leave_id = estimator.apply(X_test)

        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if (X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision node %s: %s (=%s) %s %s"
#                         % (node_id,
                                X_test.columns[feature[node_id]],
                                X_test.iloc[sample_id, feature[node_id]],
                                threshold_sign,
                                np.round(threshold[node_id],4)))
            l2r.append(
                    [X_test.columns[feature[node_id]],
                    X_test.iloc[sample_id, feature[node_id]],
                    threshold[node_id]]
            )
        return l2r


    def display_rule_per_estimator_sample_ids(estimator, X_test, sample_ids):
        """Display the decision path per tree contained in the estimator for a group of samples.
        """
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        node_indicator = estimator.decision_path(X_test)

        # we have the leaves ids reached by each sample.
        leave_id = estimator.apply(X_test)

        common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                                        len(sample_ids))

        common_node_id = np.arange(n_nodes)[common_nodes]

        print("Shared nodes %s," % (common_node_id),
                    "Shared features %s," % (
                        X_test.columns[feature[common_node_id]].values),
                    "Shared threshold %s" % (threshold[common_node_id]
                    )
        )
        print("Values:\n%s" % (X_test.iloc[sample_ids, feature[common_node_id]].values))

        return None


    def sample_rules(self):
        """Display decision path on demand if sample_id or sample_ids is not None.
        """
        l2r = []

        if self.sample_id is not None:
            print("Rules to predict sample %s" % self.sample_id)
            if 'RandomForestClassifier' in str(type(self.estimator)):
                n_trees = len(self.estimator.estimators_)
                for n in range(n_trees):
                    print("\nRules for tree %s" % n)
                    l2r = XtracTree.display_rule_per_estimator(
                            self.estimator.estimators_[n],
                            self.x_test, self.sample_id, l2r
                    )
            else:
                # the estimator is a decision tree
                # we can pass it directly to display_rule_per_estimator
                l2r = XtracTree.display_rule_per_estimator(
                            self.estimator, self.x_test,
                            self.sample_id, l2r
                )

            # we convert l2r as a dataframe
            l2r = pd.DataFrame(l2r,
                               columns=['Features',
                                        'Value Sample %s ' % self.sample_id,
                                        'Threshold']
            )

        if self.sample_ids is not None:
            print("\n\nRules to predict samples %s" % self.sample_ids)
            if 'RandomForestClassifier' in str(type(self.estimator)):
                n_trees = len(self.estimator.estimators_)
                for n in range(n_trees):
                    print("\nRules for tree %s" % n)
                    XtracTree.display_rule_per_estimator_sample_ids(
                        self.estimator.estimators_[n],
                        self.x_test, self.sample_ids
                    )
            else:
                # the estimator is a decision tree
                # we can pass it directly to display_rule_per_estimator
                XtracTree.display_rule_per_estimator_sample_ids(
                    self.estimator, self.x_test, self.sample_ids
                )

        if len(l2r): return l2r


if __name__ == '__main__':
    # we import the pkl file containing the data
    loanForExp = pkl.load(open('loanForExp.pkl','rb'), encoding='latin1')
    X_train, X_test, y_train, y_test = train_test_split(
        loanForExp.data,
        loanForExp.target,
        test_size=0.3,
        shuffle=True
    )

    exp = 2
    if exp == 1:
        estimator = (
            RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=2, max_features=2,
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=3,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
        )
    else:
        estimator = (
            DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                          criterion='gini', max_depth=10, max_features=50,
                          max_leaf_nodes=30, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort='deprecated', random_state=0, splitter='best')
        )

    estimator.fit(X_train, y_train) # model fit

    # we convert the features importance of the classifier to a df
    d = {'Features': X_train.columns,
         'Feat Imp': estimator.feature_importances_
    }
    estimatorFeatimportance = pd.DataFrame(d).sort_values(
        by='Feat Imp', ascending=False
    )

    p = XtracTree(estimator, X_train, X_test, out='estimator_decision_rules.py')
    p.build_model()

    from estimator_decision_rules import estimator_predict

    sample_ids = [0,1,2,3,4,5]
    res_from_parser = np.zeros((len(sample_ids)))
    for n in range(len(sample_ids)):
      sample_id = sample_ids[n]
      sample_proba = estimator_predict(X_test.iloc[sample_id, :])
      res_from_parser[n] = sample_proba

    d = {"sample": sample_ids,
         "Proba from XtracTree": res_from_parser,
         "Proba from DT classifier": estimator.predict_proba(X_test)[:, 1][sample_ids]}
    print(pd.DataFrame(d))

    sample_ids = np.arange(0, len(X_test[:15000]))
    res_from_parser = np.zeros((len(sample_ids)))
    for n in range(len(sample_ids)):
      sample_id = sample_ids[n]
      sample_proba = estimator_predict(X_test.iloc[sample_id, :])
      res_from_parser[n] = sample_proba

    d = {"sample": sample_ids,
         "Proba from XtracTree": res_from_parser,
         "Proba from DT classifier": estimator.predict_proba(X_test)[:, 1][sample_ids]}
    print(pd.DataFrame(d).describe())

    p = XtracTree(estimator, X_train, X_test, sample_id=6, sample_ids=[0,1,2])
    df_rules = p.sample_rules()
    print(df_rules)
