#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""xtractree_demo_rf_dt_lightgbm.py
"""
__author__ = "Jeremy Charlier"
__revised__ = "9 January 2022"
#
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import lightgbm
# PANDAS OPTIONS
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
# XTRACTREE IMPORT
from xtractree import XtracTree
#
# AVOID SKLEARN USER WARNING (DATA DEPENDENT)
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn
#
# DATA IMPORT
data=load_breast_cancer()
for item in range(len(data.feature_names)):
  data.feature_names[item]=data.feature_names[item].replace(' ', '_')
df=pd.DataFrame(data.data, columns=data.feature_names)
# DATA SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df,
    data.target,
    test_size=0.3,
    shuffle=True,
    random_state=42
)
exp = 2 # CHOOSE THE EXPERIMENTS YOU WANT TO RUN
if exp == 1:
  outfile = 'rf_xtractree.py'
  estimator = RandomForestClassifier(
    n_estimators=10, max_depth=None,
    max_features='auto', n_jobs=-1, random_state=0)
elif exp == 3:
  outfile = 'lgbm_xtractree.py'
  estimator=lightgbm.LGBMClassifier(
    max_depth=3,
    n_estimators=3,
    objective="binary"
  )
else:
  outfile = 'dt_xtractree.py'
  estimator = DecisionTreeClassifier(
    max_depth=10, max_features='auto', random_state=0)
# ENDIF
estimator.fit(X_train, y_train) # MODEL FIT
#
print('\n --- TEST 1 ---')
p = XtracTree(
  estimator,
  X_train, X_test,
  sample_id=6, sample_ids=[0,1,2],
  out=outfile)
p.build_model()
df_rules = p.sample_rules()
#
print('\n --- TEST 2 ---')
p = XtracTree(estimator, X_train, X_test, sample_id=6)
df_rules = p.sample_rules(ndecisions=3)
#
print('\n --- TEST 3 ---')
p = XtracTree(estimator, X_train, X_test, sample_id=6)
df = p.decisionsForForest(nDecisionsPerTree=5)
df
