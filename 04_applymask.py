import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy

from monai.data import *
from monai.transforms import *

sheet = pd.read_csv('dataset.csv', index_col=0)

# Create path to MCA, check if file exists
def find_mca(mr):
    path = os.path.join("../Sources/CTA", f"{mr}_CTA_MCA.nii.gz")
    if os.path.isfile(path):
         print('found it')
    else: # some files diverge from the agreed name convention
        path = os.path.join("../Sources/CTA", f"{mr}_MCA.nii.gz")
        if os.path.isfile(path):
             print('found it alternatively')
        else: 
             path = False
             print("did not find at all")
    return path
sheet["path_mca"] = [find_mca(x) for x in sheet.index]
#sheet['fex'] = [os.path.isfile(x) for x in sheet.path_mca]
#miss_mca = sheet[sheet['fex']==True]
#sheet = sheet[sheet['fex']==True]

# Output
outpath = "Dataset/masked"
if not os.path.exists(outpath):
        os.makedirs(outpath)
sheet["path_out"] = [os.path.join(outpath, f"{x}.nii.gz") 
                     for x in sheet.index]

warning = list()
for index, elem in enumerate(sheet.index):
    
    patient = sheet.path.tolist()[index]
    mca = sheet.path_mca.tolist()[index]
    out = sheet.path_out.tolist()[index]
    print(index, patient)

    if os.path.isfile(out):
         print("file already exist, jump")
         continue

    p = nib.load(patient)
    paffine = p.affine
    m = nib.load(mca)
    psc = p.header['pixdim'][1:4]
    p = p.get_fdata()
    print(p.shape, psc)
    msc = m.header['pixdim'][1:4]
    m = m.get_fdata()
    print(m.shape, msc)

    p = p * m
    if (p.shape != m.shape): warning.append(index)
    empty_header = nib.Nifti1Header()
    pii = nib.Nifti1Image(p, paffine, empty_header)
    nib.save(pii, sheet['path_out'].tolist()[index])

print(warning)
sheet.to_csv("04_applymask_output.csv")
 
