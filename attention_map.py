# Converted from attention_map.ipynb

# %% Cell 1
import torch
from torchvision import models
from torch.autograd import Function
import matplotlib.pyplot as plt 
from medcam import medcam
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
import pandas as pd 

# %% Cell 2
# Load model
import monai
resnet_model=monai.networks.nets.resnet50(
    pretrained=False,
    spatial_dims=3,
    n_input_channels=1, 
    num_classes=1)
checkpoint = torch.load("Collection/version_3033490/c-epoch=167-val_loss_epoch=tensor(0.5776, device='cuda_0').ckpt")
weights_dict = {k.replace('_model.', ''): v for k, v in checkpoint['state_dict'].items()}
model_dict = resnet_model.state_dict()
model_dict.update(weights_dict)
resnet_model.load_state_dict(model_dict)
resnet_model = resnet_model.to(device) 

patientlist = pd.read_csv('img-preds.csv')
patientlist = list(patientlist.MRN)

# %% Cell 3
# plotting function
def plot_slices(original, attention, slice, threshold, patient):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
    
    # Rotate the original and attention maps by 90 degrees
    original = np.rot90(original, k=-1)
    attention = np.rot90(attention, k=-1)
    
    image_stack = original[:, :, slice:(slice+10)]
    axes.imshow(original[:, :, slice], cmap='gray')
    #axes.set_facecolor('black')  
    #fig.patch.set_facecolor('black') 
    attention_stack = attention[:, :, slice:(slice+10)]
    attention_mip = np.max(attention_stack, axis=2)
    
    min_value = np.min(attention_mip)
    max_value = np.max(attention_mip)
    threshold = ((max_value - min_value) * 5/10) + min_value 
    print(f"Min: {min_value}, Max: {max_value}, Calculated Threshold: {threshold}")
    masked_attention = np.ma.masked_less(attention_mip, threshold)
    
    cmap = plt.cm.get_cmap('jet')
    
    im = axes.imshow(masked_attention, cmap=cmap, alpha=0.7, vmin=np.min(attention), vmax=np.max(attention))
    
    axes.set_title(f"Slice {slice}")
    axes.axis('off')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    #cbar_ax.ax.yaxis.set_tick_params(color='white')  # Set color of colorbar ticks
    #cbar_ax.set_label('Colorbar Label', color='white')  # Set color of colorbar label
    fig.colorbar(im, cax=cbar_ax)
    

    plt.savefig(f'attentionmap_{patient}_{slice}.png')
    plt.show()

# %% Cell 4
# create attention map
#p = 'MR95457'
#layer_name = 'layer4'
#backend_type = 'ggcam'
def create_attentionmap(p, layer_name, backend_type, slice):
    model = medcam.inject(resnet_model, output_dir='attention_maps', backend=backend_type, layer=layer_name, label='best', save_maps=True)
    print(p, backend_type, layer_name)
    input_image = f"Dataset/voxel_standard_fullhead/{p}.nii.gz"
    nifti_img = nib.load(input_image)
    img_data = nifti_img.get_fdata()
    tensor_img = torch.from_numpy(img_data)
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.unsqueeze(0)
    target_class = 0
    print(tensor_img.shape)
    tensor_img = tensor_img.float()
    tensor_img = tensor_img.to(device)
    #print(tensor_img.device)

    output = model(tensor_img)

    input_image = f"attention_maps/{layer_name}/attention_map_0_0_0.nii.gz"
    if backend_type == "gbp":
         input_image = "attention_maps/attention_map_0_0_0.nii.gz"
    nifti_img = nib.load(input_image)
    attention_map = nifti_img.get_fdata() 
    print('Attention_Map shape:', attention_map.shape)

    # MIP over 10 Slides
    normalized_attention_map = attention_map
    original_image = img_data
    original_image[original_image > 100] = 100
    original_image[original_image < 0] = 0
    
    threshold = 155
    plot_slices(original_image, normalized_attention_map, slice, threshold=threshold, patient=p)

# %% Cell 5
create_attentionmap('MR5821734', 'layer4', 'ggcam', 70)

# %% Cell 6
for p in patientlist:
    create_attentionmap(p, 'layer3', 'ggcam')

# %% Cell 7
# Printing slides as they are

resize = False
slices = [50, 60, 70]
threshold = 140


if resize == True: 
    # Assuming you have the attention map as a NumPy array called 'attention_map'
    original_image_size = (128, 128, 128)  # Size of the original 3D image
    resized_attention_map = zoom(attention_map, original_image_size / np.array(attention_map.shape), order=1)
    normalized_attention_map = resized_attention_map
else:
    normalized_attention_map = attention_map
original_image = img_data
original_image[original_image > 100] = 100
original_image[original_image < 0] = 0

def plot_slices(original, attention, slices, threshold):
    num_slices = len(slices)
    fig, axes = plt.subplots(1, num_slices, figsize=(5*num_slices, 5))
    
    for i, slice_idx in enumerate(slices):
        # Replace 'axis' with the appropriate axis for the third dimension of your image data
        axes[i].imshow(original[:, :, slice_idx], cmap='gray')
        
        masked_attention = np.ma.masked_less(attention[:, :, slice_idx], threshold)
        cmap = plt.cm.get_cmap('jet')
        
        im = axes[i].imshow(masked_attention, cmap=cmap, alpha=0.7, vmin=np.min(attention), vmax=np.max(attention))

        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
plot_slices(original_image, normalized_attention_map, slices, threshold=threshold)

# %% Cell 8
am_nifti = nib.Nifti1Image(normalized_attention_map, affine=np.eye(4))
output_path = "attention_maps/output.nii.gz"
nib.save(am_nifti, output_path)
