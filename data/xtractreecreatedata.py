# -*- coding: utf-8 -*-
"""xtractreeCreateData.ipy
"""

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
from sklearn.utils import Bunch
import pickle as pkl

# pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

def importcsv(fileName):
    loan_data = pd.read_csv(fileName).fillna(0)

    dropList = ["id", "member_id", "sub_grade", "emp_title", "emp_length", "url",
             "desc", "purpose", "title", "zip_code", "earliest_cr_line",
             "initial_list_status", "last_pymnt_d", "next_pymnt_d",
             "last_credit_pull_d", "verification_status_joint",
             "sec_app_earliest_cr_line", "hardship_type", "hardship_reason",
             "hardship_status", "deferral_term", "hardship_start_date",
             "payment_plan_start_date", "hardship_length", "hardship_dpd",
             "hardship_loan_status", "orig_projected_additional_accrued_interest",
             "debt_settlement_flag_date", "settlement_status", "settlement_date",
             "settlement_amount", "settlement_percentage", "settlement_term",
             "mths_since_last_delinq", "mths_since_last_record",
             "mths_since_last_major_derog", "annual_inc_joint", "dti_joint",
             "mths_since_recent_revol_delinq", "mths_since_recent_revol_delinq",
             "revol_bal_joint", "sec_app_inq_last_6mths", "sec_app_mort_acc",
             "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
             "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
             "sec_app_collections_12_mths_ex_med",
             "sec_app_mths_since_last_major_derog", "hardship_amount",
             "hardship_payoff_balance_amount", "hardship_last_payment_amount",
             "grade", "addr_state", "application_type",
             "hardship_flag", "disbursement_method", "debt_settlement_flag",
             "verification_status", "issue_d", "hardship_end_date"]
    loan_data = loan_data.drop(columns=dropList)

    loan_data = loan_data[(loan_data['loan_status']=='Charged Off') | (
        loan_data['loan_status']=='Default') | (
        loan_data['loan_status']=='Fully Paid')]

    loan_data = loan_data.rename(columns={"loan_status": "target"})
    loan_data.term.loc[loan_data.term == ' 36 months'] = 36
    loan_data.term.loc[loan_data.term == ' 60 months'] = 60

    loan_data.replace('Fully Paid', 0, inplace=True)
    loan_data.replace('Default', 1, inplace=True)
    loan_data.replace('Charged Off', 1, inplace=True)
    loan_data = loan_data.replace('MORTGAGE', 0)
    loan_data = loan_data.replace('OWN', 1)
    loan_data = loan_data.replace('RENT', 2)
    loan_data = loan_data.replace('ANY', 3)
    loan_data = loan_data.replace('NONE', 4)
    loan_data = loan_data.replace('OTHER', 5)
    loan_data['pymnt_plan'] = loan_data['pymnt_plan'].replace('n', 0)

    return loan_data


df = importcsv("loan.csv")

sns.set(style="whitegrid")
sns.barplot(
    x="target", y="loan_amnt",
    data=df, ci=None, estimator=np.median)

sns.barplot(
    x="target", y="int_rate",
    data=df, estimator=np.median)

sns.barplot(
    x="target", y="annual_inc",
    data=df, estimator=np.median)

sns.barplot(
    x="target", y="tot_cur_bal",
    data=df, ci=None, estimator=np.median)

# we put the dataset in bunch
loanForExp = Bunch(
    target_names = ['Fully Paid', 'Default'],
    target=df['target'].values,
    data=df.drop(columns='target')
)
