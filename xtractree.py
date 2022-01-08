#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""xtractree.py
"""
#
import numpy as np
import pandas as pd
from collections import OrderedDict
#
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
#
# pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
#
#
class XtracTree:
  def __init__(self, estimator, x_train, x_test, 
               sample_id=None, sample_ids=None, out=None):
    self.estimator = estimator
    self.x_train = x_train
    self.x_test = x_test
    self.sample_id = sample_id
    self.sample_ids = sample_ids
    self.out = out
  # end of function __init__
  #
  def __getstate__(self):
    return self.__dict__.copy()
  # end of function __getstate__
  #
  def trees_to_dataframe(self):
    """Parse the fitted model and return in an easy-to-read pandas DataFrame.
    The returned DataFrame has the following columns.
      - ``tree_index`` : int64, which tree a node belongs to. 0-based, so a
        value of ``6``, for example, means "this node is in the 7th tree".
      - ``node_depth`` : int64, how far a node is from the root of the tree.
        The root node has a value of ``1``, its direct children are ``2``, etc.
      - ``node_index`` : str, unique identifier for a node.
      - ``left_child`` : str, ``node_index`` of the child node to the left of
        a split. ``None`` for leaf nodes.
      - ``right_child`` : str, ``node_index`` of the child node to the 
        right of a split. ``None`` for leaf nodes.
      - ``parent_index`` : str, ``node_index`` of this node's parent.
        ``None`` for the root node.
      - ``split_feature`` : str, name of the feature used for splitting.
        ``None`` for leaf nodes.
      - ``split_gain`` : float64, gain from adding this split to the tree.
        ``NaN`` for leaf nodes.
      - ``threshold`` : float64, value of the feature used to decide which
        side of the split a record will go down. ``NaN`` for leaf nodes.
      - ``decision_type`` : str, logical operator describing how to compare
        a value to ``threshold``.
        For example, ``split_feature = "Column_10", threshold = 15,
        decision_type = "<="`` means that
        records where ``Column_10 <= 15`` follow the left side of the split,
        otherwise follows the right side of the split. ``None`` for leaf nodes.
      - ``missing_direction`` : str, split direction that missing values
        should go to. ``None`` for leaf nodes.
      - ``missing_type`` : str, describes what types of values
        are treated as missing.
      - ``value`` : float64, predicted value for this leaf node, multiplied
        by the learning rate.
      - ``weight`` : float64 or int64, sum of hessian (second-order
        derivative of objective), summed over observations that fall
        in this node.
      - ``count`` : int64, number of records in the training data that
        fall into this node.
    Returns
    -------
    result : pandas DataFrame
        Returns a pandas DataFrame of the parsed model.
    """
    if self.num_trees() == 0:
      msg='There are no trees in this Booster and thus nothing to parse'
      raise LightGBMError(msg)
    # ENDIF
    def _is_split_node(tree):
        return 'split_index' in tree.keys()
    # END OF FUNCTION _is_split_node
    #
    def create_node_record(tree, node_depth=1, tree_index=None,
                            feature_names=None, parent_node=None):
        def _get_node_index(tree, tree_index):
            tree_num = f'{tree_index}-' if tree_index is not None else ''
            is_split = _is_split_node(tree)
            node_type = 'S' if is_split else 'L'
            # if a single node tree it won't have `leaf_index` so return 0
            node_num = tree.get('split_index' if is_split else 'leaf_index', 0)
            return f"{tree_num}{node_type}{node_num}"
        # END OF FUNCTION _get_node_index
        #
        def _get_split_feature(tree, feature_names):
            if _is_split_node(tree):
                if feature_names is not None:
                    feature_name = feature_names[tree['split_feature']]
                else:
                    feature_name = tree['split_feature']
            else:
                feature_name = None
            return feature_name
        # END OF FUNCTION _get_split_feature
        #
        def _is_single_node_tree(tree):
            return set(tree.keys()) == {'leaf_value'}
        # END OF FUNCTION _is_single_node_tree
        #
        # Create the node record, and populate universal data members
        node = OrderedDict()
        node['tree_index'] = tree_index
        node['node_depth'] = node_depth
        node['node_index'] = _get_node_index(tree, tree_index)
        node['left_child'] = None
        node['right_child'] = None
        node['parent_index'] = parent_node
        node['split_feature'] = _get_split_feature(tree, feature_names)
        node['split_gain'] = None
        node['threshold'] = None
        node['decision_type'] = None
        node['missing_direction'] = None
        node['missing_type'] = None
        node['value'] = None
        #node['weight'] = None
        node['count'] = None
        #
        # Update values to reflect node type (leaf or split)
        if _is_split_node(tree):
            node['left_child'] = _get_node_index(tree['left_child'], tree_index)
            node['right_child'] = _get_node_index(tree['right_child'], tree_index)
            node['split_gain'] = tree['split_gain']
            node['threshold'] = tree['threshold']
            node['decision_type'] = tree['decision_type']
            node['missing_direction'] = 'left' if tree['default_left'] else 'right'
            node['missing_type'] = tree['missing_type']
            node['value'] = tree['internal_value']
            #node['weight'] = tree['internal_weight']
            node['count'] = tree['internal_count']
        else:
            node['value'] = tree['leaf_value']
            if not _is_single_node_tree(tree):
                #node['weight'] = tree['leaf_weight']
                node['count'] = tree['leaf_count']

        return node
    # END OF FUNCTION create_node_record
    #
    def tree_dict_to_node_list(
        tree, node_depth=1, tree_index=None,
        feature_names=None, parent_node=None):
      node = create_node_record(
        tree,
        node_depth=node_depth,
        tree_index=tree_index,
        feature_names=feature_names,
        parent_node=parent_node
      )
      res = [node]
      if _is_split_node(tree):
        # traverse the next level of the tree
        children = ['left_child', 'right_child']
        for child in children:
          subtree_list = tree_dict_to_node_list(
            tree[child],
            node_depth=node_depth + 1,
            tree_index=tree_index,
            feature_names=feature_names,
            parent_node=node['node_index'])
          # In tree format, "subtree_list" is a list of node records (dicts),
          # and we add node to the list.
          res.extend(subtree_list)
      return res
    # END OF FUNCTION tree_dict_to_node_list
    #
    model_dict = self.dump_model()
    feature_names = model_dict['feature_names']
    model_list = []
    for tree in model_dict['tree_info']:
      model_list.extend(
        tree_dict_to_node_list(
          tree['tree_structure'],
          tree_index=tree['tree_index'],
          feature_names=feature_names
        )
      )
    # ENDFOR
    return pd.DataFrame(model_list, columns=model_list[0].keys())
  # END OF FUNCTION trees_to_dataframe
  #
  def build_dtree_rules(estimator, n_trees, x_train, l2w):
    """Extract decision rules with splitting thresholds and probabilities.
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
    # begin while loop
    while len(stack) > 0:
      node_id, parent_depth = stack.pop()
      node_depth[node_id] = parent_depth + 1
      # if we have a test node
      if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
      else:
        is_leaves[node_id] = True
      # end if
    # end while loop
    bgn="        if state == "
    # begin for loop
    for i in range(n_nodes):
      if is_leaves[i]:
        irow = bgn+"%s: return %s\n" % (
          i, ((value[i][0,1] / weighted_n_node_samples[i]) / n_trees)
          )
      else:
        irow = bgn+"%s: state = (%s if x['%s'] <= %s else %s)\n" % (
          i, 
          children_left[i],
          x_train.columns[feature[i]],
          threshold[i],
          children_right[i]
        )
      # end if
      l2w.append(irow)
    # end for loop
    return l2w
  # end of function build_dtree_rules
  #
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
    #
    if 'LGBM' in str(self.estimator):
      dfmodel=XtracTree.trees_to_dataframe(self.estimator._Booster)
      nrows=len(dfmodel)
      nsamples=dfmodel.iloc[0]["count"]
      n_trees=self.estimator.n_estimators
      for itree in range(n_trees):
        l2w.append("  if num_tree == %s:\n" % itree)
        dftree=dfmodel[dfmodel.tree_index==itree]
        node_depth=np.unique(dftree.node_depth)
        state,cnt=0,0
        l2w.append("    state=%s\n" % state)
        msg="    if state=="
        for idepth in node_depth:
          indx=dftree.node_depth==idepth
          tmpdf=dftree[indx]
          for nrow in range(len(tmpdf)):
            irow=tmpdf.iloc[nrow]
            if irow.decision_type=="<=":
              msg="    if state=="
              l2w.append(msg+"%s: state=(%s if x['%s'] %s %s else %s)\n" % (
                state,
                state+1+cnt,
                irow.split_feature,
                irow.decision_type,
                irow.threshold,
                state+2+cnt)
              )
              cnt+=1
            else:
              l2w.append(msg+'%s: return %s\n' % (
                state, irow.value)
              )
              cnt-=1
            # ENDIF
            state+=1
          # ENDFOR
        # ENDFOR
      # ENDFOR
    else:
      l2w.append("    if num_tree == %s:\n" % 0)
      l2w.append("        state = %s\n" % 0)
      if 'RandomForestClassifier' in str(type(self.estimator)):
        n_trees = len(self.estimator.estimators_)
        # begin for loop
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
          # end if
        # end for loop
      else:
        # the estimator is a decision tree
        # we can pass it directly to build_tree_rules
        n_trees = 1
        l2w = XtracTree.build_dtree_rules(
          self.estimator, 
          n_trees, self.x_train, l2w)
      # ENDIF
    # ENDIF
    # we write at the bottom the predict rule for the fixed model
    l2w.append("\n\ndef estimator_predict(x):\n")
    # we initialize the proba values at 0
    l2w.append("    predict = 0.0\n")
    l2w.append("    for i in range(%s):\n" % n_trees)
    l2w.append("        predict += estimator_tree(x, i)\n")
    if 'LGBM' in str(self.estimator):
      l2w.append("    predict=(1/(1+np.exp(-predict)))\n")
    # ENDIF
    l2w.append("    return predict\n")
    if self.out is not None:
      with open(self.out, 'w') as the_file:
        the_file.write("".join(l2w))
        the_file.close()
      # end with
    else:
      print(l2w)
    # end if
    return None
  # end of function build_model
  #
  def display_rule_per_estimator(estimator, X_test, sample_id, l2r):
    """Display the decision path per tree contained in the estimator
    for 1 sample.
    """
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    node_indicator = estimator.decision_path(X_test)
    # we have the leaves ids reached by each sample
    leave_id = estimator.apply(X_test)
    node_index = node_indicator.indices[
      node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    # begin for loop
    for node_id in node_index:
      if leave_id[sample_id] == node_id:
        continue
      # end if
      if (X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
      else:
        threshold_sign = ">"
      # end if
      print("decision node %s: %s (=%s) %s %s"
        % (node_id,
          X_test.columns[feature[node_id]],
          X_test.iloc[sample_id, feature[node_id]],
          threshold_sign,
          np.round(threshold[node_id],4))
      )
      l2r.append(
        [X_test.columns[feature[node_id]], 
        X_test.iloc[sample_id, feature[node_id]],
        threshold[node_id]]
    )
    # end for loop
    return l2r
  # end of function display_rule_per_estimator
  #
  def display_rule_per_estimator_sample_ids(estimator, X_test, sample_ids):
    """Display the decision path per tree contained in the estimator
    for a group of samples.
    """
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    node_indicator = estimator.decision_path(X_test)
    # we have the leaves ids reached by each sample.
    leave_id = estimator.apply(X_test)
    common_nodes = (
      node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    )
    common_node_id = np.arange(n_nodes)[common_nodes]
    print("Shared nodes %s,\n" % (common_node_id),
      "Shared features %s,\n" % (X_test.columns[feature[common_node_id]].values), 
      "Shared threshold %s" % (threshold[common_node_id])
    )
    print("Values:\n%s\n" % (
      X_test.iloc[sample_ids, feature[common_node_id]].values)
    )
    return None
  # end of function display_rule_per_estimator_sample_ids
  #
  def sample_rules(self):
    """Display decision path on demand if sample_id or sample_ids
    is not None.
    """
    if 'LGBM' in str(self.estimator):
      print('sample_rules for lightgbm not yet implemented.')
      return None
    # ENDIF
    l2r = []
    if self.sample_id is not None:
      print("Rules to predict sample %s" % self.sample_id)
      if 'RandomForestClassifier' in str(type(self.estimator)):
        n_trees = len(self.estimator.estimators_)
        # begin for loop
        for n in range(n_trees):
          print("\nRules for tree %s" % n)
          l2r = XtracTree.display_rule_per_estimator(
            self.estimator.estimators_[n],
            self.x_test, self.sample_id, l2r
          )
        # end for loop
      else:
        # the estimator is a decision tree
        # we can pass it directly to display_rule_per_estimator
        l2r = XtracTree.display_rule_per_estimator(
          self.estimator, self.x_test, 
          self.sample_id, l2r
        )
      # end if
      # we convert l2r as a dataframe
      l2r = pd.DataFrame(
        l2r, 
        columns=[
          'Features', 
          'Value Sample %s ' % self.sample_id, 
          'Threshold']
      )
    # end if
    if self.sample_ids is not None:
      print("\n\nRules to predict samples %s" % self.sample_ids)
      if 'RandomForestClassifier' in str(type(self.estimator)):
        n_trees = len(self.estimator.estimators_)
        # begin for loop
        for n in range(n_trees):
          print("\nRules for tree %s" % n)
          XtracTree.display_rule_per_estimator_sample_ids(
            self.estimator.estimators_[n],
            self.x_test, self.sample_ids
          )
        # end for loop
      else:
        # the estimator is a decision tree
        # we can pass it directly to display_rule_per_estimator
        XtracTree.display_rule_per_estimator_sample_ids(
          self.estimator, self.x_test, self.sample_ids
        )
      # end if
    if len(l2r): return l2r
  # end of function sample_rules
# end of class XtracTree
#
#
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
