import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd

df = r"dataset.csv"
df = pd.read_csv(df, index_col=0)
df = df[['mRS','label']]
# drop faulty images

clinc = pd.read_excel("LVOthrombectomy_yale2021_mod_030.xlsx")
clinc = clinc[["MRN",'TICI', 'LKW-CTA', 'CTA-angio']]
#clinc = clinc.dropna()
clinc = clinc.set_index('MRN')

df = pd.merge(left=df, right=clinc, left_index=True, right_index=True, how="inner")
df.rename(
    columns={'LKW-CTA':'LKN_CTA_time',
     'CTA-angio':'CTA_Angio_time'}, inplace=True
)

# because the number is by date and not by hours.
#df.LKN_CTA_time = df.LKN_CTA_time * 24
#df.LKN_CTA_time.clip(lower=0, inplace=True)
#df.CTA_Angio_time = df.CTA_Angio_time * 24
#df.CTA_Angio_time.clip(lower=0, inplace=True)
# also no na values

# two images are faulty
df.drop(['MR1391245','MR1670035'], inplace=True)

# trainVal and testing split set
X_trainVal, X_test, y_trainVal, y_test = train_test_split(
    df[['TICI','LKN_CTA_time','CTA_Angio_time']], df.label, test_size=0.16, random_state=42)
# train and val kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ktrain_index = list()
kval_index = list()
for i, (train_index, val_index) in enumerate(skf.split(X_trainVal, y_trainVal)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    ktrain_index.append(train_index)
    print(f"  Test:  index={val_index}")
    kval_index.append(val_index)


# 
for k in range(5):
    X_train = X_trainVal.iloc[ktrain_index[k]]
    y_train = y_trainVal[ktrain_index[k]]
    X_val = X_trainVal.iloc[kval_index[k]]
    y_val = y_trainVal[kval_index[k]]
    X_train['label'] = y_train
    X_train['train'] = 1
    X_val['label'] = y_val
    X_val['train'] = 0
    X_train = X_train.append(X_val)
    X_train.to_excel(f'fold_og_{k}.xlsx')

X_test['label'] = y_test
X_test.to_excel('fold_og_test.xlsx')


