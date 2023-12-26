import os
import numpy as np
import pandas as pd

# import mastersheet from above level, only use the 3m_mRS excel sheet
sheetmrs = pd.read_excel('../dataset_mod.xlsx')
#mastersheet = pd.read_excel("LVOthrombectomy_yale2021.xlsx")
#mastersheet = mastersheet.loc[:,["MRN", "D_mRS", "3m_mRS"]]
mastersheet = sheetmrs.copy()
#mastersheet.set_index('MRN', inplace=True)

# checking where data is missing and do not use these files
missing_3m = mastersheet[mastersheet['3m_mRS'].isna()].copy()
missing_3m.loc[:,'3m_mRS'] = missing_3m.loc[:,'D_mRS']
filled_3m = missing_3m['3m_mRS']

mastersheet['3m_mRS'].update(filled_3m)
mastersheet = mastersheet[['MRN','3m_mRS','TICI','LKN_CTA_time', 'CTA_Angio_time', 'headImage500', 'mask500']]
mastersheet = mastersheet.dropna()

# INPUT PATH
# register all files from the folder
file_loc = r'../../Sources/CTA'
regist = list()
files = os.listdir(file_loc)

for f in files:
    name_long = f
    mrn = f.split(sep='_')[0]
    path = file_loc + r"/" + name_long
    r = {
            "mrn":mrn,
            "full_name":name_long,
            "path":path}
    regist.append(r)
regist = pd.DataFrame(data=regist)

# only select the right type of file of every patient
regist_sel = regist[regist['full_name'].str.contains('_head.nii.gz')]
#regist_sel = regist[regist['full_name'].str.contains('_CTA.nii.gz')]

# inner join regist_sel with mastersheet, now only the patients that have both 3mRS and the right image gets written in the data
df = mastersheet.set_index('MRN').join(regist_sel.set_index('mrn'), how='inner')
df = df.rename(columns={"3m_mRS": "mRS"})

# create label
def writelabel(mrs):
    if mrs >= 3: l = 1
    else: l = 0
    return l 
df['label'] = [writelabel(x) for x in df.mRS]
df.label.value_counts()
#df = df[df.label < 2]
df = df[["mRS",'label','TICI','LKN_CTA_time', 'CTA_Angio_time', 'full_name','path']]
df
df.to_csv('dataset.csv')