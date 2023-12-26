# MILESTONE FREEZE

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import monai
from monai.config import print_config
from monai.data import DataLoader, list_data_collate
from monai.transforms import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import RocCurveDisplay

########### CONFIG ############
SEED = 42
SWITCH_TRAIN = True
SWITCH_YALE = not (SWITCH_TRAIN)

# choose if time stamps should be normed and scaled
NORMSCALE = 0

# Fold
FOLD = 0
OG = True
VAL_AS_TEST = False

MODEL = "LR"
# 'RF', 'LR', Bayesian, GradientBoost

print('##### PARAMETERS #####')
print('Seed: ', SEED)
print('Model ', MODEL)

####################################

# SWITCHES TO TURN OFF PART OF THE CODE
monai.utils.set_determinism(seed=SEED, additional_settings=None)
np.random.seed(SEED)
# ------------------ 
       
# ---- LOAD MODEL 
def normandscale_clinical(TICI, LKWCTA, CTAANGIO, mode):
    if mode == 1:
        x = LKWCTA.to_numpy()
        lkwcta_n = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = CTAANGIO.to_numpy()
        ctaangio_n = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = TICI.to_numpy()
        tici_n = x / 4
        result = np.column_stack((tici_n, lkwcta_n, ctaangio_n))
    else: 
        lkwcta_n  = LKWCTA.to_numpy()
        ctaangio_n = CTAANGIO.to_numpy()
        tici_n = TICI.to_numpy()
        result = np.column_stack((tici_n, lkwcta_n, ctaangio_n))
    return result

print('---------')
# not using all folds, just 3/5

if OG: 
    filename = f'fold_og_{FOLD}.xlsx'
    dataset = pd.read_excel(filename, index_col=0)
    dataset = dataset.dropna()
    filename_test = f'fold_og_test.xlsx'
    testset = pd.read_excel(filename_test, index_col=0)
    testset = testset.dropna()

else: 
    filename = f'fold_{FOLD}.xlsx'
    dataset = pd.read_excel(filename, index_col=0)
    dataset = dataset.dropna()
    filename_test = f'fold_test.xlsx'
    testset = pd.read_excel(filename_test, index_col=0)
    testset = testset.dropna()
print('FILENAME FOLD: ', filename)
print('FILENAME TEST: ', filename_test)
if VAL_AS_TEST:
    dftrain = dataset[dataset.train == 1]
    dfval = dataset[dataset.train == 0]
    
    dataset = dftrain
    testset = dfval
    
print('Training Patients: ', len(dataset))
print('Validation Patients: ', len(testset))
X = dataset[['TICI','LKN_CTA_time','CTA_Angio_time']]
y = dataset.label

if MODEL == "RF":
    clf = RandomForestClassifier(
        max_depth=100, 
        random_state=SEED)
    clf.fit(X, y)
    preds = clf.predict_proba(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
    RocCurveDisplay.from_predictions(testset.label, preds.T[1])
elif MODEL == 'LR':
    clf = LogisticRegression(random_state=SEED)
    clf.fit(X, y)
    preds = clf.predict_proba(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
    RocCurveDisplay.from_predictions(testset.label, preds.T[1])
elif MODEL == "Bayesian":
    clf = BayesianRidge().fit(X,y)
    preds = clf.predict(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
    RocCurveDisplay.from_predictions(testset.label, preds)
    
elif MODEL == "GradientBoost":
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=11, random_state=0).fit(X, y)
    preds = clf.predict_proba(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
    RocCurveDisplay.from_predictions(testset.label, preds.T[1])
elif MODEL == 'Lasso':
    from sklearn import linear_model
    reg = linear_model.Lasso(alpha=0.1).fit(X, y)
    preds = reg.predict(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
    preds


# Logistic Regression with Cross Validation

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
preds = clf.predict(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
probabilities = clf.predict_proba(testset[['TICI','LKN_CTA_time','CTA_Angio_time']])
RocCurveDisplay.from_predictions(testset.label, probabilities.T[1])

print(classification_report(testset.label, preds, labels=[0,1]))
testset['preds'] = preds
testset['probabilities'] = probabilities.T[1]
export_preds = testset[['label','preds', 'probabilities']]
export_preds
export_preds.to_csv('treat-preds.csv')

