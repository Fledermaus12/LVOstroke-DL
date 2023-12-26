import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import optim, nn
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import RocCurveDisplay, roc_auc_score
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import pandas as pd
import monai
from monai.config import print_config
from monai.data import DataLoader, list_data_collate
from monai.transforms import *
import warnings
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
warnings.filterwarnings('ignore')
with torch.no_grad(): torch.cuda.empty_cache()

SEED = 42

# choose if it should use full head CT or just masked images
MASKED = False


OG = True

torch.manual_seed(SEED)
monai.utils.set_determinism(seed=SEED, additional_settings=None)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
np.random.seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print('---------')




if MASKED: 
    folder_path = "Dataset/voxel_standard_masked"
else:
    folder_path = "Dataset/voxel_standard_fullhead"
print('FOLDER_PATH: ', folder_path)

print('---------')

# Load Test
filename = f'fold_og_test.xlsx'
dftest = pd.read_excel(filename, index_col=0)
dftest.reset_index(inplace=True)
dftest.rename({'index': 'MRN'}, axis=1, inplace=True)
mastersheet = pd.read_excel('LVOthrombectomy_yale2021_mod_030.xlsx')
mastersheet.rename({'Unnamed: 0': 'MRN'}, axis=1, inplace=True)
dftest = dftest.merge(mastersheet, on='MRN', how='left')
dftest = dftest[['MRN','TICI_x', 'LKW-CTA', 'LKW-angio', 'NIHSS', 'label', 'Age', 'Sex']]
dftest = dftest.dropna()
dftest.set_index('MRN', inplace=True)

dftest['img'] = [os.path.join(folder_path, f"{x}.nii.gz") 
            for x in dftest.index]
dftest = dftest.dropna()

model_output = pd.DataFrame({"MRN": dftest.index})
model_output['labels'] = dftest['label'].tolist()


# ------------------ 
# Load TrainVal File
# Fold
for FOLD in range(5):
    if OG: 
        filename = f'fold_og_{FOLD}.xlsx'
    else: 
        filename = f'fold_{FOLD}.xlsx'
    print('FILENAME: ', filename)
    dataset = pd.read_excel(filename, index_col=0)
    dataset.reset_index(inplace=True)
    dataset.rename({'index': 'MRN'}, axis=1, inplace=True)
    dataset = dataset.merge(mastersheet, on='MRN', how='left')
    dataset = dataset[['MRN','TICI_x', 'LKW-CTA', 'LKW-angio', 'NIHSS', 'label', 'Age', 'Sex', 'train']]
    dataset = dataset.dropna()
    dataset.set_index('MRN', inplace=True)

    ######## LOAD MODEL #####
    # Modellarchitektur definieren
    resnet_model=monai.networks.nets.resnet50(
        pretrained=False,
        spatial_dims=3,
        n_input_channels=1, 
        num_classes=1)
    CHECKPOINTS = [
            "Collection/version_3033490/c-epoch=167-val_loss_epoch=tensor(0.5776, device='cuda_0').ckpt",
            "Collection/version_3033525/c-epoch=167-val_loss_epoch=tensor(0.6086, device='cuda_0').ckpt",
            "Collection/version_3033567/c-epoch=281-val_loss_epoch=tensor(0.5935, device='cuda_0').ckpt",
            "Collection/version_3033675/c-epoch=69-val_loss_epoch=tensor(0.6515, device='cuda_0').ckpt",
            "Collection/version_3033705/c-epoch=133-val_loss_epoch=tensor(0.6156, device='cuda_0').ckpt"
        ]
    ckpt = CHECKPOINTS[FOLD]
    checkpoint = torch.load(ckpt)
    weights_dict = {k.replace('_model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_dict = resnet_model.state_dict()
    model_dict.update(weights_dict)
    resnet_model.load_state_dict(model_dict)

    # delete last layer
    resnet = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    #
    #
    # TURN FOLLOWING ON FOR OUTPUT 1: 
    #
    #
    resnet = resnet_model 

    # Set the model to evaluation mode
    resnet.eval()

    val_files =  [{"img": img,
                "label": label,
                "clinical":clinical} for 
                img, label, clinical in 
                zip(dfval['img'],
                    dfval.label,
                    dfval[['TICI_x','LKW-CTA','LKW-angio', 'NIHSS', 'Age', 'Sex']].to_numpy())]
    test_files =  [{"img": img,
                    "label": label,
                    "clinical":clinical} for 
                    img, label, clinical in 
                    zip(dftest['img'],
                        dftest.label,
                        dftest[['TICI_x','LKW-CTA','LKW-angio','NIHSS', 'Age', 'Sex']].to_numpy())]
    transforms_valtest = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ToTensord(keys=['img'])

        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=transforms_valtest)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    test_ds = monai.data.Dataset(data=test_files, transform=transforms_valtest)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    all_features = []
    all_labels = []
    i = 0
    for batch in val_loader:
        # Move the input data to the training device
        img, clinical, labels = batch['img'].to(device), batch['clinical'].to(device), batch['label'].to(device)
        
        with torch.no_grad():
            # Assuming you have already defined and loaded the pre-trained ResNet model
            resnet = resnet.to(device)
            
            features = resnet(img)
            features = features.view(1, -1)
            
            clinical_data = clinical
            
            combined_features = torch.cat((features, clinical_data), dim=1)
            all_features.append(combined_features.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
        i = i + 1
        print('VALSET FEATURE', i)

    X = np.vstack(all_features)
    y = np.vstack(all_labels)



    # Test set
    all_features = []
    all_labels = []
    i = 0
    for batch in test_loader:
        # Move the input data to the training device
        img, clinical, labels = batch['img'].to(device), batch['clinical'].to(device), batch['label'].to(device)
        
        with torch.no_grad():
            # Assuming you have already defined and loaded the pre-trained ResNet model
            resnet = resnet.to(device)
            
            features = resnet(img)
            features = features.view(1, -1)
            
            clinical_data = clinical
            
            combined_features = torch.cat((features, clinical_data), dim=1)
            all_features.append(combined_features.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
        i = i + 1
        print('TEST FEATURES', i)

    Xtest = np.vstack(all_features)
    ytest = np.vstack(all_labels)

    clf = LogisticRegression(random_state=42).fit(X, y)
    preds = clf.predict_proba(Xtest)
    print('LR: ', roc_auc_score(ytest.T[0], preds.T[1]))
    model_output[f'output_{FOLD}'] = preds.T[1]

    model_output.to_csv('img_treat_clin-preds.csv')
    
####
model_output = pd.read_csv('img_treat_clin-preds.csv', index_col=0)
print(model_output)
model_output.set_index('MRN', inplace=True)
labels = model_output.pop('labels')
def checkoutput(n, thr):
    if n >= thr:
        return 1
    else: return 0

from sklearn.metrics import classification_report
# Mean vote
model_output['mean_prob'] = model_output.mean(axis=1)
RocCurveDisplay.from_predictions(labels, model_output.mean_prob)

model_output['decision'] = [checkoutput(i, 0.50) for i in model_output.mean_prob]
print(classification_report(labels, model_output.decision))
model_output['labels'] = labels
model_output.to_csv('img_treat_clin-preds.csv')

    

    