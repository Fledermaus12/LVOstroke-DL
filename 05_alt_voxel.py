import register, convert
import pandas as pd
import nibabel as nib
import os
import ants
from tempfile import mkstemp

MASKED = False

if MASKED: 
    sheet = pd.read_csv('04_applymask_output.csv', index_col=0)
    sheet['path'] = sheet.path_out
    sheet.drop(['path_mca', 'path_out'], axis=1, inplace=True)

    outpath = "Dataset/voxel_standard_masked"
    if not os.path.exists(outpath): os.makedirs(outpath)
else:
    sheet = pd.read_csv('dataset.csv', index_col=0) 
    sheet.drop(['full_name'], axis=1, inplace=True)

    outpath = "Dataset/voxel_standard_fullhead"
    if not os.path.exists(outpath): os.makedirs(outpath)

TEMPLATE_PATH = 'scct_unsmooth_SS_0.01_128x128x128.nii.gz'
template = ants.image_read(TEMPLATE_PATH, pixeltype = 'float')

def save_as_nii(img, patient, outpath):
    """
    Convert an ANTsImage to a Nibabel image
    """
    filename = patient+".nii.gz"
    pth = os.path.join(outpath, filename)
    print(pth)
    img.to_filename(pth)

i = 0
for idx, row in sheet.iterrows():
    print(f'{i}/{len(sheet)}: {row.name}')
    original_image = nib.load(row['path'])
    original_header = original_image.header
    original_affine = original_image.affine
    image = convert.nii2ants(original_image)

    image, transforms = register.rigid(template, image)

    save_as_nii(image,row.name,outpath)
    i = i+1
