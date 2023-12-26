import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_recall_curve
import random
random.seed(123)
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import norm
import pandas as pd 
import nibabel as nib
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import precision_score, accuracy_score

from scipy.stats import mannwhitneyu
from monai.data import *
from monai.transforms import *
from sklearn.metrics import roc_curve

table = pd.read_excel('LVOthrombectomy_yale2021_mod_030.xlsx')
table = table[['MRN','D_mRS', '3m_mRS', 'Sex', 'NIHSS','Age', 'TICI']]

# load Predictions
imgclin7 = pd.read_csv('img_treat-preds.csv')
imgt2 = pd.read_csv('img-preds.csv')

img_only_prob = imgt2.mean_prob
img_only_decision = imgt2.decision

img_clin_prob = imgclin7.mean_prob
img_clin_decision = imgclin7.decision
truth = imgclin7.labels


# threshold finding accuracy
threshold = []
accuracy = []

for p in np.unique(img_only_prob):
    threshold.append(p)
    img_only_prob_temp = (img_only_prob >= p).astype(int)
    accuracy.append(accuracy_score(truth,img_only_prob_temp))

plt.scatter(threshold,accuracy)
plt.xlabel("Threshold")
plt.ylabel("Balanced accuracy")
plt.show()
print(threshold[np.argmax(accuracy)])
print(np.max(accuracy))

# threshold finding accuracy
threshold = []
accuracy = []

for p in np.unique(img_clin_prob):
    threshold.append(p)
    img_clin_prob_temp = (img_clin_prob >= p).astype(int)
    accuracy.append(accuracy_score(truth,img_clin_prob_temp))

plt.scatter(threshold,accuracy)
plt.xlabel("Threshold")
plt.ylabel("accuracy")
plt.show()
print(threshold[np.argmax(accuracy)])
print(np.max(accuracy))

imgclin7['img_only_decision'] = img_only_decision
imgclin7['img_only_prob'] = img_only_prob
imgclin7['img_clin_decision'] = img_clin_decision
imgclin7['img_clin_prob'] = img_clin_prob
df = imgclin7[['MRN','labels','img_only_decision','img_only_prob','img_clin_decision','img_clin_prob']]

def checkoutput(n, thr):
    if n >= thr:
        return 1
    else: return 0

# threshold from threshold_opt file
df['img_only_decision'] = [checkoutput(i, 0.53) for i in df['img_only_prob']]
df['img_clin_decision'] = [checkoutput(i, 0.54) for i in df['img_clin_prob']]



# ----
modelchoice = 'img_only_decision'
mask = df['labels'] == df[modelchoice]
print(len(df[mask]))

modelchoice = 'img_clin_decision'
mask = df['labels'] == df[modelchoice]
print(len(df[mask]))





modelchoice = 'img_clin_decision'


#------------ mRS ----------------------
original_dataset = pd.read_csv('dataset.csv')
original_dataset.rename(columns={'Unnamed: 0':'MRN'}, inplace=True)
mRS_list = original_dataset[['MRN', 'mRS']]
total = df.copy()
total = total.merge(mRS_list, on='MRN', how='left')
print(total.mRS.describe())
total_vc = total.mRS.value_counts().sort_index()
print(total_vc.loc[:'2.0'].sum()/total_vc.sum())
mask = df['labels'] == df[modelchoice]
tntp = df[mask]
tntp = tntp.merge(mRS_list, on='MRN', how='left')
tntp_vc = tntp.mRS.value_counts().sort_index()
mask = df['labels'] != df[modelchoice]
fnfp = df[mask]
fnfp = fnfp.merge(mRS_list, on='MRN', how='left')
fnfp_vc = fnfp.mRS.value_counts().sort_index()
print(modelchoice)
print('group tntp')
print(tntp.mRS.describe())
print('Proportion mRS < 3: ', tntp_vc.loc[:'2.0'].sum()/tntp_vc.sum())
print()
print('group fnfp')
print(fnfp.mRS.describe())
print('Proportion mRS < 3: ', fnfp_vc.loc[:'2.0'].sum()/fnfp_vc.sum())
statistic, p_value = mannwhitneyu(fnfp.mRS, tntp.mRS, alternative='two-sided')
print(statistic)
print(p_value)

#------------ Sex ----------------------
total = df.copy()
total = total.merge(table[['MRN','Sex']], on='MRN', how='left')
total
mask = df['labels'] == df[modelchoice]
tntp = df[mask]
tntp = tntp.merge(table[['MRN','Sex']], on='MRN', how='left')
tntp
tntp_vc = tntp.Sex.value_counts().sort_index()
tntp_vc
mask = df['labels'] != df[modelchoice]
fnfp = df[mask]
fnfp = fnfp.merge(table[['MRN','Sex']], on='MRN', how='left')
fnfp_vc = fnfp.Sex.value_counts().sort_index()
fnfp_vc
print(modelchoice)
print('group tntp')
print(tntp.Sex.describe())
print()
print('group fnfp')
print(fnfp.Sex.describe())

print('TNTP')
count = len(tntp[tntp['Sex']==2])
print('Sex Male: ', count)
print('Total: ', len(tntp))
print('Relative: ',count/len(tntp))

print('FNFP')
count = len(fnfp[fnfp['Sex']==2])
print('Sex Male: ', count)
print('Total: ', len(fnfp))
print('Relative: ',count/len(fnfp))

# Chi Square Test -> categorical 
var1 = tntp.Sex
var2 = fnfp.Sex
contingency_table = pd.crosstab(var1, var2)
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(p_value)

#------------ NIHSS ----------------------
total = df.copy()
total = total.merge(table[['MRN','NIHSS']], on='MRN', how='left')

mask = df['labels'] == df[modelchoice]
tntp = df[mask]
tntp = tntp.merge(table[['MRN','NIHSS']], on='MRN', how='left')
mask = df['labels'] != df[modelchoice]
fnfp = df[mask]
fnfp = fnfp.merge(table[['MRN','NIHSS']], on='MRN', how='left')
print(modelchoice)
print('group tntp')
print(tntp.NIHSS.describe())
print()
print('group fnfp')
print(fnfp.NIHSS.describe())


statistic, p_value = mannwhitneyu(fnfp.NIHSS, tntp.NIHSS, alternative='two-sided')
print(statistic)
print(round(p_value,4))

#------------ Age ----------------------
total = df.copy()
total = total.merge(table[['MRN','Age']], on='MRN', how='left')

mask = df['labels'] == df[modelchoice]
tntp = df[mask]
tntp = tntp.merge(table[['MRN','Age']], on='MRN', how='left')
mask = df['labels'] != df[modelchoice]
fnfp = df[mask]
fnfp = fnfp.merge(table[['MRN','Age']], on='MRN', how='left')
print(modelchoice)
print('group tntp')
print(tntp.Age.describe())
print()
print('group fnfp')
print(fnfp.Age.describe())
statistic, p_value = mannwhitneyu(fnfp.Age, tntp.Age, alternative='two-sided')
print(statistic)
print(round(p_value,4))
#print(stats.ttest_ind(fnfp.Age, tntp.Age))


#------------ mTICI ----------------------
total = df.copy()
total = total.merge(table[['MRN','TICI']], on='MRN', how='left')
mask = df['labels'] == df[modelchoice]
tntp = df[mask]
tntp = tntp.merge(table[['MRN','TICI']], on='MRN', how='left')
mask = df['labels'] != df[modelchoice]
fnfp = df[mask]
fnfp = fnfp.merge(table[['MRN','TICI']], on='MRN', how='left')
print(modelchoice)
print('group tntp')
print(tntp.TICI.describe())
#print('Proportion mRS < 3: ', tntp_vc.loc[:'2.0'].sum()/tntp_vc.sum())
print()
print('group fnfp')
print(fnfp.TICI.describe())
statistic, p_value = mannwhitneyu(fnfp.TICI, tntp.TICI, alternative='two-sided')
print(statistic)
print(round(p_value,4))

