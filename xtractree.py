import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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
        """
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
                # irow = "%sif state == %s:\n%sprint(state)\n%sreturn %s\n" % ((
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
        """
        to complete the description
        """
        # we write the first line of the file
        # we store the content of the file to be written in l2w
        l2w = ["import numpy as np\n"]
        l2w.append("\n")
        l2w.append("def rf_tree(x, num_tree):\n")
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
        l2w.append("\n\ndef rf_predict(x):\n")

        # we initialize the proba values at 0
        l2w.append("    predict = 0.0\n")
        l2w.append("    for i in range(%s):\n" % n_trees)
        l2w.append("        predict += rf_tree(x, i)\n")
        l2w.append("    return predict\n")

        if self.out is not None:
            with open(self.out, 'w') as the_file:
                the_file.write("".join(l2w))
                the_file.close()
        else:
            print(l2w)

        return None


    def display_rule_per_estimator(estimator, X_test, sample_id, l2r):
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
                        % (node_id,
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
        """
        Finish the description of the function.
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


# __main__():
breast_cancer = load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df['target'] = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['target']),
    df['target'],
    test_size=0.3,
    random_state=42
)

# rf_trivial = RandomForestClassifier(
#     n_estimators=5, criterion='gini',
#     max_depth=10, min_samples_split=2,
#     min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#     max_features=4, max_leaf_nodes=None,
#     min_impurity_decrease=0.0, min_impurity_split=None,
#     bootstrap=True, oob_score=False,
#     n_jobs=None, random_state=None,
#     verbose=0, warm_start=False,
#     class_weight=None, ccp_alpha=0.0, max_samples=None
# )

rf_trivial = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)

rf_trivial.fit(X_train, y_train)

# we convert the features importance of the classifier to a df
d = {'Features': X_train.columns,
     'Feat Imp': rf_trivial.feature_importances_
}
rf_trivial_featimportance = pd.DataFrame(d).sort_values(
    by='Feat Imp', ascending=False
)

# we build the model
p = XtracTree(rf_trivial, X_train, X_test, out='rf_decision_rules.py')
p.build_model()

# we compare the prediction of the py file with the model
from rf_decision_rules import rf_predict

sample_ids = [0,1,2,3,4,5]
res_from_parser = np.zeros((len(sample_ids)))
for n in range(len(sample_ids)):
  sample_id = sample_ids[n]
  sample_proba = rf_predict(X_test.iloc[sample_id, :])
  res_from_parser[n] = sample_proba

d = {"sample": sample_ids,
     "Proba from parser": res_from_parser,
     "Proba from classifier": rf_trivial.predict_proba(X_test)[:, 1][sample_ids]}
pd.DataFrame(d)

# we extract and display rules
p = XtracTree(rf_trivial, X_train, X_test, sample_id=0, sample_ids=[0,1,2])
df_rules = p.sample_rules()
