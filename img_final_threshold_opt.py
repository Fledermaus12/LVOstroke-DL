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
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = 42

# choose if it should use full head CT or just masked images

# Fold
fold = 0

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


file = f'fold_og_{fold}.xlsx'
print('FILENAME: ', file)

# Test     
df = pd.read_excel(file, index_col=0)
df = df.dropna()
df = df[df.train == 0]


folder_path = "Dataset/voxel_standard_fullhead"
print('FOLDER_PATH: ', folder_path)


df['img'] = [os.path.join(folder_path, f"{x}.nii.gz") 
        for x in df.index]

# put data in dataloader 
transforms = Compose(
    [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ToTensord(keys=['img'])

    ]
)
files = [{"img": img,
            "label": label} for 
            img, label in 
            zip(df.img,
                df.label)]

ds = monai.data.Dataset(data=files, transform=transforms)
loader = DataLoader(ds, batch_size=1, num_workers=4)

# Prepare a variable to insert the data there later
model_output = pd.DataFrame({"MRN": df.index})
model_output['labels'] = df['label'].tolist()

CHECKPOINT = [
    "Collection/version_3033490/c-epoch=167-val_loss_epoch=tensor(0.5776, device='cuda_0').ckpt",
    "Collection/version_3033525/c-epoch=167-val_loss_epoch=tensor(0.6086, device='cuda_0').ckpt",
    "Collection/version_3033567/c-epoch=281-val_loss_epoch=tensor(0.5935, device='cuda_0').ckpt",
    "Collection/version_3033675/c-epoch=69-val_loss_epoch=tensor(0.6515, device='cuda_0').ckpt",
    "Collection/version_3033705/c-epoch=133-val_loss_epoch=tensor(0.6156, device='cuda_0').ckpt"
]

ckpt = CHECKPOINT[fold]
checkpoint = torch.load(ckpt)
weights_dict = {k.replace('_model.', ''): v for k, v in checkpoint['state_dict'].items()}
model_dict = resnet_model.state_dict()
model_dict.update(weights_dict)
resnet_model.load_state_dict(model_dict)

resnet_model.to(device)
resnet_model.eval()
# Create Train files
loader_output = []
loader_labels = []
i = 0
for batch in loader:
    # Move the input data to the training device
    img, labels = batch['img'].to(device), batch['label'].to(device)
    with torch.no_grad():
        outputs = resnet_model(img)
        outputs = torch.sigmoid(outputs)
        loader_output.append(outputs.detach().cpu().numpy())
        loader_labels.append(labels.detach().cpu().numpy())
    print('PREDICT ', i)
    i += 1
    #if i == 0: break

# put the output together
loader_output = np.concatenate(loader_output, axis=0)
loader_labels = np.stack(loader_labels)
model_output[f'output_{fold}'] = loader_output.T[0]
RocCurveDisplay.from_predictions(loader_labels, loader_output)

model_output

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(model_output.labels,model_output[f'output_{fold}'],drop_intermediate=False)
plt.scatter(thresholds,np.abs(fpr+tpr-1))
plt.xlabel("Threshold")
plt.ylabel("|FPR + TPR - 1|")
plt.show()
thresholds[np.argmin(np.abs(fpr+tpr-1))]

# balanced accuracy
threshold = []
accuracy = []

for p in np.unique(model_output[f'output_{fold}']):
  threshold.append(p)
  y_pred = (model_output[f'output_{fold}'] >= p).astype(int)
  accuracy.append(accuracy_score(model_output.labels,y_pred))

plt.scatter(threshold,accuracy)
plt.xlabel("Threshold")
plt.ylabel("accuracy")
plt.show()
threshold[np.argmax(accuracy)]