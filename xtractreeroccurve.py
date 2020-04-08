#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xtractreeROCCurve.ipynb
"""

import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_curve,
                             auc)

# pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

loanForExp = pkl.load(open('loanForExp.pkl','rb'), encoding='latin1')
X_train, X_test, y_train, y_test = train_test_split(
    loanForExp.data,
    loanForExp.target,
    test_size=0.3,
    shuffle=True
)

if False:
    if exp == 1:
        # we perform a quick grid search to optimize the classification
        params = {'n_estimators':[2,5,10],
                  'max_depth':[2,5,10],
                  'max_features':[10,30,50],
                  'min_samples_split': [2,4,10]}

    if exp == 2:
        # we perform a quick grid search to optimize the classification
        params = {'max_depth':[2,5,10,50],
                  'max_features':[2,5,10,50],
                  'max_leaf_nodes':[10,30,50,100,500]}

    gsearch = GridSearchCV(estimator,
                          param_grid=params,
                          n_jobs=-1, # use all processors
                          cv=None, # use the default 5-fold cross validation
                          iid=False, # loss minimized is total loss per sample
                          verbose=10
                          )
    gsearch.fit(X_train, y_train)

    print('\ngsearch.best_params_:\n', gsearch.best_params_)
    print('gsearch.best_estimator_:\n', gsearch.best_estimator_)


# we plot the ROC AUC curves
# gsearch.best_params_:
#  {'max_depth': 5, 'max_features': 50, 'max_leaf_nodes': 10}
# gsearch.best_estimator_:
#  DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#                        max_depth=5, max_features=50, max_leaf_nodes=10,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort='deprecated',
#                        random_state=0, splitter='best')

dtestimator = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                      criterion='gini', max_depth=2, max_features=50,
                      max_leaf_nodes=10, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort='deprecated', random_state=0, splitter='best')
dtestimator.fit(X_train, y_train)

# gsearch.best_params_:
#  {'max_depth': 2, 'max_features': 30, 'min_samples_split': 2, 'n_estimators': 5}
# gsearch.best_estimator_:
#  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=2, max_features=30,
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=5,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
rfestimator = (
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=5, max_features=30,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
)
rfestimator.fit(X_train, y_train)

fprDT, tprDT, _ = roc_curve(
    y_test, dtestimator.predict_proba(X_test)[:,1])
fprRF, tprRF, _ = roc_curve(
    y_test, rfestimator.predict_proba(X_test)[:,1])

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(
    fprDT, tprDT,
    label='DT( AUC: %s \u00B1 0.01)' % (
        np.round(auc(fprDT, tprDT), 3)
    )
)
plt.plot(
    fprRF, tprRF,
    label='RF( AUC: %s \u00B1 0.01)' % (
        np.round(auc(fprRF, tprRF), 3)
    )
)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
