# Converted from img_final.ipynb

# %% Cell 1
# Load modules and data
import os
import torch
from torch import optim, nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.tensorboard
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchmetrics
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split, StratifiedKFold
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import pandas as pd
import monai
from monai.config import print_config
from monai.data import DataLoader, list_data_collate
from monai.transforms import *
import warnings
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Cell 2

SEED = 42

# choose if it should use full head CT or just masked images
MASKED = False

# Fold
FOLD = 0
OG = True

# determinism
torch.manual_seed(SEED)
monai.utils.set_determinism(seed=SEED, additional_settings=None)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
np.random.seed(SEED)

# Modellarchitektur definieren
resnet_model=monai.networks.nets.resnet50(
    pretrained=False,
    spatial_dims=3,
    n_input_channels=1, 
    num_classes=1)

if OG: 
    testfile = 'fold_og_test.xlsx'
else: 
    testfile = 'fold_test.xlsx'
print('FILENAME: ', testfile)

# Test     
dataset_test = pd.read_excel(testfile, index_col=0)
dataset_test = dataset_test.dropna()


if MASKED: 
    folder_path = "Dataset/voxel_standard_masked"
else:
    folder_path = "Dataset/voxel_standard_fullhead"
print('FOLDER_PATH: ', folder_path)


dataset_test['img'] = [os.path.join(folder_path, f"{x}.nii.gz") 
        for x in dataset_test.index]

# %% Cell 3
# put data in dataloader 
transforms_valtest = Compose(
    [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ToTensord(keys=['img'])

    ]
)
test_files = [{"img": img,
            "label": label} for 
            img, label in 
            zip(dataset_test.img,
                dataset_test.label)]

test_ds = monai.data.Dataset(data=test_files, transform=transforms_valtest)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

# Prepare a variable to insert the data there later
model_output = pd.DataFrame({"MRN": dataset_test.index})
model_output['labels'] = dataset_test['label'].tolist()

# %% Cell 4
CHECKPOINT = [
    "Collection/version_3033490/c-epoch=167-val_loss_epoch=tensor(0.5776, device='cuda_0').ckpt",
    "Collection/version_3033525/c-epoch=167-val_loss_epoch=tensor(0.6086, device='cuda_0').ckpt",
    "Collection/version_3033567/c-epoch=281-val_loss_epoch=tensor(0.5935, device='cuda_0').ckpt",
    "Collection/version_3033675/c-epoch=69-val_loss_epoch=tensor(0.6515, device='cuda_0').ckpt",
    "Collection/version_3033705/c-epoch=133-val_loss_epoch=tensor(0.6156, device='cuda_0').ckpt"
]

# %% Cell 5
for j in range(5):
    ckpt = CHECKPOINT[j]
    checkpoint = torch.load(ckpt)
    weights_dict = {k.replace('_model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_dict = resnet_model.state_dict()
    model_dict.update(weights_dict)
    resnet_model.load_state_dict(model_dict)

    resnet_model.to(device)
    resnet_model.eval()
    # Create Train files
    test_output = []
    test_labels = []
    i = 0
    for batch in test_loader:
        # Move the input data to the training device
        img, labels = batch['img'].to(device), batch['label'].to(device)
        with torch.no_grad():
            outputs = resnet_model(img)
            outputs = torch.sigmoid(outputs)
            test_output.append(outputs.detach().cpu().numpy())
            test_labels.append(labels.detach().cpu().numpy())
        print('PREDICT ', i)
        i += 1
        #if i == 0: break
    
    # put the output together
    test_output = np.concatenate(test_output, axis=0)
    test_labels = np.stack(test_labels)
    model_output[f'output_{j}'] = test_output.T[0]
    RocCurveDisplay.from_predictions(test_labels, test_output)
 

# %% Cell 6
model_output.to_csv('Img-t2-modelprediction-2.csv')

# %% Cell 7
model_output = pd.read_csv('Img-t2-modelprediction-2.csv', index_col=0)
print(model_output)
model_output.set_index('MRN', inplace=True)
labels = model_output.pop('labels')
def checkoutput(n, thr):
    if n >= thr:
        return 1
    else: return 0

model_output

# %% Cell 8

# Majority
for i in range(5):
    model_output[f'output_{i}_class'] = [checkoutput(i, 0.50) for i in model_output[f'output_{i}']]
def voting(df, MRN):
    l = list()
    for i in range(5):
        l.append(df.loc[MRN, f'output_{i}_class'])
    if sum(l) > 2: return 1
    else: return 0
model_output['decision'] = [voting(model_output,mrn) for mrn in model_output.index]
print(model_output)
score = roc_auc_score(labels, model_output.decision)
print(f"ROC AUC: {score:.4f}")
print(classification_report(labels, model_output.decision))
RocCurveDisplay.from_predictions(labels, model_output.decision)

# %% Cell 9

# Mean vote
model_output['mean_prob'] = model_output.mean(axis=1)
RocCurveDisplay.from_predictions(labels, model_output.mean_prob)

model_output['decision'] = [checkoutput(i, 0.50) for i in model_output.mean_prob]
print(classification_report(labels, model_output.decision))
model_output['labels'] = labels
model_output.to_csv('img-t2-modelprediction-mean.csv')

# %% Cell 10
