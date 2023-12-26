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
       
# ---- LOAD FILES 
mastersheet = pd.read_excel('LVOthrombectomy_yale2021_mod_030.xlsx')
mastersheet.rename({'Unnamed: 0': 'MRN'}, axis=1, inplace=True)

 
filename = f'fold_og_{FOLD}.xlsx'
dataset = pd.read_excel(filename, index_col=0)
dataset.reset_index(inplace=True)
dataset.rename({'index': 'MRN'}, axis=1, inplace=True)
dataset = dataset.merge(mastersheet, on='MRN', how='left')
dataset = dataset[['MRN','TICI_x', 'LKW-CTA', 'LKW-angio', 'NIHSS', 'label', 'Age', 'Sex', 'train']]
dataset.set_index('MRN', inplace=True)
dataset = dataset.dropna()
dataset

filename_test = f'fold_og_test.xlsx'
dftest = pd.read_excel(filename_test, index_col=0)
dftest.reset_index(inplace=True)
dftest.rename({'index': 'MRN'}, axis=1, inplace=True)
dftest = dftest.merge(mastersheet, on='MRN', how='left')
dftest = dftest[['MRN','TICI_x', 'LKW-CTA', 'LKW-angio', 'NIHSS', 'label', 'Age', 'Sex']]
dftest = dftest.dropna()
dftest.set_index('MRN', inplace=True)
dftest



print('FILENAME FOLD: ', filename)
print('FILENAME TEST: ', filename_test)

    
print('Training Patients: ', len(dataset))
print('Validation Patients: ', len(dftest))
X = dataset[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']]
y = dataset.label




# Logistic Regression with Cross Validation

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
preds = clf.predict(dftest[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']])
preds
probabilities = clf.predict_proba(dftest[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']])
RocCurveDisplay.from_predictions(dftest.label, probabilities.T[1])

print(classification_report(dftest.label, preds, labels=[0,1]))

dftest['preds'] = preds
dftest['probabilities'] = probabilities.T[1]
export_preds = dftest[['label','preds', 'probabilities']]
export_preds.to_csv('treat_clin-preds.csv')
export_preds



clf = RandomForestClassifier(
    max_depth=500, 
    random_state=SEED)
clf.fit(X, y)
preds = clf.predict(dftest[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']])
probabilities = clf.predict_proba(dftest[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']])
RocCurveDisplay.from_predictions(dftest.label, probabilities.T[1])
print(classification_report(dftest.label, preds, labels=[0,1]))

'''
if MODEL == "RF":
    clf = RandomForestClassifier(
        max_depth=100, 
        random_state=SEED)
    clf.fit(X, y)
    preds = clf.predict_proba(dftest[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']])
    RocCurveDisplay.from_predictions(dftest.label, preds.T[1])
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
'''