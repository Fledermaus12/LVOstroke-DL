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
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

SEED = 42
SWITCH_TRAIN = True
SWITCH_YALE = not (SWITCH_TRAIN)

DROPOUT = 0

# choose if it should use full head CT or just masked images
MASKED = False

MAX_EPOCH = 300 
BATCH_SIZE = 6

# Optimizer
LEARNING_RATE = 0.000001
WD = 0.001

# Fold
FOLD = 0
OG = True

print('##### PARAMETERS #####')
print('Seed: ', SEED)
print('DROPOUT ', DROPOUT)
print('MAX_EPOCH ', MAX_EPOCH)
print('BATCH_SIZE ', BATCH_SIZE)
print('LEARNING_RATE ', LEARNING_RATE)
print('WD ', WD)

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
weights=torch.load('resnet_50.pth')
weights_dict = {k.replace('module.', ''): v for k, v in weights['state_dict'].items()}
model_dict = resnet_model.state_dict()
model_dict.update(weights_dict)
resnet_model.load_state_dict(model_dict)

class CombinedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = resnet_model
        # Defining loss
        self.loss = nn.BCEWithLogitsLoss()
        self.train_auc = torchmetrics.classification.AUROC(task="binary")
        self.val_auc = torchmetrics.classification.AUROC(task="binary")
        
    def forward(self, x):
        x = self._model(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self._model.parameters(), lr=LEARNING_RATE, weight_decay=WD)
        #scheduler = MultiStepLR(opt, milestones=[61, 250], gamma=0.1)
        return opt


    def training_step(self, batch, batch_idx):
        img, labels = batch['img'], batch['label']
        outputs = self.forward(img)
    
        # Loss
        #l1 = torch.nn.functional.one_hot(labels, num_classes=2)
        l1 = labels.reshape(-1,1).float()
        loss = self.loss(outputs, l1)
        
        # output for AUC
        outputs = torch.sigmoid(outputs)
        outputs = torch.tensor(outputs)
        self.train_auc(outputs, labels)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        img, labels = batch['img'], batch['label']
        outputs = self.forward(img)
        
        # Loss
        l1 = labels.reshape(-1,1).float()
        loss = self.loss(outputs, l1)
        if self.current_epoch < 2: print('Loss: ', loss)
        # outputL for AUC
        #print("Outputs: ", outputs)
        #print("Labels: ", labels)
        
        outputs = torch.sigmoid(outputs)
        outputs = torch.tensor(outputs)
        self.val_auc(outputs, labels)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
# ---- LOAD MODEL 



if OG: 
    filename = f'fold_og_{FOLD}.xlsx'
else: 
    filename = f'fold_{FOLD}.xlsx'
print('FILENAME: ', filename)
dataset = pd.read_excel(filename, index_col=0)


if MASKED: 
    folder_path = "Dataset/voxel_standard_masked"
else:
    folder_path = "Dataset/voxel_standard_fullhead"
print('FOLDER_PATH: ', folder_path)

print('---------')
dataset['img'] = [os.path.join(folder_path, f"{x}.nii.gz") 
            for x in dataset.index]
dftrain = dataset[dataset.train == 1]
dfval = dataset[dataset.train == 0]


train_files = [{"img": img,
                "label": label} for 
                img, label in 
                zip(dftrain.img,
                    dftrain.label)]
val_files =  [{"img": img,
                "label": label} for 
                img, label in 
                zip(dfval.img,
                    dfval.label)]


transforms = Compose(
    [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        #ThresholdIntensityd(
        #    keys=['img'], 
        #    threshold=100,
        #    above=False,
        #    cval=100),
        #ThresholdIntensityd(
        #    keys=['img'], 
        #    threshold=0,
        #    above=True,
        #    cval=0),
        RandRotated(
            keys=['img'], 
            range_x=[-0.2, 0.2],
            range_y=[-0.2, 0.2],
            range_z=[-0.2, 0.2],
            prob=0.3),
        RandZoomd(
            keys=['img'],
            prob=0.3,
            min_zoom=0.8,
            max_zoom=1.2),
        RandFlipd(
            keys=['img'],
            prob=0.2
        ),
        RandAffined(
            keys=['img'],
            prob=0.3,
            shear_range=(0.3,0.3),
            translate_range=(0.3,0.3)
        ),
        ToTensord(keys=['img'])
    ]
)
transforms.set_random_state(seed=SEED)
transforms_valtest = Compose(
    [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ToTensord(keys=['img'])

    ]
)
# testing the transforms and save an image
if False:
    # only for testing the image quality
    transforms_try = Compose(
    [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ThresholdIntensityd(
            keys=['img'], 
            threshold=80,
            above=False,
            cval=80),
        ThresholdIntensityd(
            keys=['img'], 
            threshold=0,
            above=True,
            cval=0),
        RandRotated(
            keys=['img'], 
            range_x=[-0.2, 0.2],
            range_y=[-0.2, 0.2],
            range_z=[-0.2, 0.2],
            prob=0.3),
        RandZoomd(
            keys=['img'],
            prob=0.3,
            min_zoom=0.8,
            max_zoom=1.2),
        RandFlipd(
            keys=['img'],
            prob=0.2
        ),
        RandAffined(
            keys=['img'],
            prob=0.2,
            shear_range=(0.3,0.3),
            translate_range=(0.3,0.3)
        ),
        SaveImaged(keys=['img'],
                   output_ext=".nii.gz",
                   output_dir="monai_try",
                   writer="NibabelWriter")
    ])
    transforms_try.set_random_state(seed=SEED)
    transforms_try(train_files[:1])

train_ds = monai.data.Dataset(data=train_files, transform=transforms)
val_ds = monai.data.Dataset(data=val_files, transform=transforms_valtest)
#test_ds = monai.data.Dataset(data=test_files, transform=transforms_valtest)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
#test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

# ----
# Training 
if SWITCH_TRAIN:
    model = CombinedModel()
    val_checkpoint = ModelCheckpoint(
        save_top_k=10, 
        #every_n_epochs=5,
        monitor='val_auc',
        mode='max',
        filename="c-{epoch:02d}-{val_auc:.2f}")
    loss_checkpoint = ModelCheckpoint(
        save_top_k=10, 
        #every_n_epochs=5,
        monitor='val_loss_epoch',
        mode='min',
        filename="c-{epoch:02d}-{val_loss_epoch}")
    latest_checkpoint = ModelCheckpoint(
        save_top_k=3, 
        every_n_epochs=50,
        monitor='epoch',
        mode='max',
        filename="c-{epoch:02d}")
    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=MAX_EPOCH, 
        #accumulate_grad_batches=1,
        check_val_every_n_epoch=2, 
        #strategy = DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=5,
        num_sanity_val_steps=-1,
        callbacks=[val_checkpoint, loss_checkpoint, latest_checkpoint])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.cuda.empty_cache()


# Test with Yale Test Set  
if SWITCH_YALE:
    print("-------------- TESTING ---------------- ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = 'Collection/version_2966429/c-epoch=57-val_auc=0.77.ckpt'
    CHECKPOINT = "Collection/version_2966429/c-epoch=127-val_loss_epoch=tensor(0.5720, device='cuda_0').ckpt"
    


    model = CombinedModel.load_from_checkpoint(CHECKPOINT)
    if OG:
        testfile = 'fold_og_test.xlsx'
    else:
        testfile = 'fold_test.xlsx'
    dataset_test = pd.read_excel(testfile, index_col=0)
    dataset_test['img'] = [os.path.join(folder_path, f"{x}.nii.gz") 
            for x in dataset_test.index]

    test_files = [{"img": img,
                "label": label} for 
                img, label in 
                zip(dataset_test.img,
                    dataset_test.label)]
    test_ds = monai.data.Dataset(data=test_files, transform=transforms_valtest)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
    model.to(device)
    model.eval()

    # Create Train files
    test_output = []
    test_labels = []
    i = 0
    for batch in test_loader:
        # Move the input data to the training device
        img, labels = batch['img'].to(device), batch['label'].to(device)
        with torch.no_grad():
            outputs = model(img)
            test_output.append(outputs.detach().cpu().numpy())
            test_labels.append(labels.detach().cpu().numpy())
        print(i)
        i += 1
        #if i == 0: break
    
    # put the output together
    test_output = np.concatenate(test_output, axis=0)
    test_labels = np.stack(test_labels)


    from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
    RocCurveDisplay.from_predictions(test_labels, test_output)
    model_output = pd.DataFrame({"labels":test_labels.T[0], 
                            "output":test_output.T[0]})
    #fpr, tpr, thresholds = roc_curve(test_labels, test_output, pos_label=1)
    roc_auc_score(test_labels, test_output, average=None)
    #model_output.to_csv('modeleval_yale-m1(img)+m2(tici+t1+t2).csv')
