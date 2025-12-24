# %%
import cv2
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm 

import clean_dataset as CD
import construct_dataset as ConsD
import fine_tune_model as FTM
import test as TS
import segment_dataset as SD
import recombination as RCB
import compute_metrics as CM
import frac_volumes as FV 


correct_colors = {
    1: [255, 1, 13], # stroma
    0: [1, 1, 1],    # background
    2: [1, 90, 255]  # cellular
}

# %%
# 🎨 Manual tiles annotation

dim = 512   # dimension of the images    
#root_path = "C:/Users/augus/Desktop/PORTFOLIO GITHUB/Dataset DenseNet fine tuning/"     
#root_path = "C:/Users/augus/Desktop/tissues_segmentation/Test/"   
root_path = "C:/Users/augus/Desktop/Chinese data/manual segmentation/"            
source_path = root_path + "Source tiles"                                       # path to source images 
source_resized_path = root_path + "Source resized tiles"                       # path to save resized tiles
mask_path = root_path + "Mask tiles"                                           # path to manual annotation masks 
semantic_desti = root_path + "Semantic tiles"                                  # path for cleaned masks 
path_df_seg = root_path + "Pixel classifier/"                                  # path to save the dataset as a dataframe
classifier_path = root_path + "Pixel classifier/fine_tuned_densenet_FAP.pth"   # path to save the classifier once trained 

# %%
# 🧹 Data refinement 

CD.construct_dataset(source_path, source_resized_path, mask_path, semantic_desti, dim, correct_colors)

# %%
# 🗃️ PyTorch dataset integration

df_DN, weights_class = ConsD.dataset_df(source_resized_path, semantic_desti)
df_DN.to_pickle(path_df_seg + "dataset_semantic_segmentation")

print(f"Class weights: {weights_class}")

# %%
# 🚀 Data loader creation

train_loader, val_loader = ConsD.init_dataloader(df_DN, batch_size = 8, shuffle = True)

# %%
# 🏗️ Neural architecture design

## Performed when necessary in the files

# %%
# ⛳ Model fine-tuning

patience = 10
num_epochs = 20
metrics_name = 'training_metrics.csv' # saved in the current folder 

FTM.model_FineTune(patience, num_epochs, train_loader, val_loader, dim, weights_class, metrics_name)

# %%
# 👁️ Segmentation visualization 

img_test_path = ""
semantic_test_path = ""

TS.visualize_seg(img_test_path, semantic_test_path, correct_colors)

# %%
# 🌎 Full-scale dataset segmentation

# >> paths inner/border source tiles
tissue_dataset_path = root_path + "inner_tiles/FAP"
tissue_dataset_border_path = root_path + "border_tiles/FAP"   
#tissue_dataset_path = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient2/inner_tiles/HE/" + letter
#tissue_dataset_border_path = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient2/border_tiles/HE/" + letter
#tissue_dataset_path = root_path + "Source resized tiles"

# >> destination inner/border tiles 
seg_dataset_path = root_path + "inner_tiles/Neoplastic/"
seg_dataset_border_path = root_path + "border_tiles/Neoplastic/" 
#seg_dataset_path = root_path + "Predicted tiles 2"

# >> Segment inner & border tiles 
SD.segment_dataset(tissue_dataset_path, seg_dataset_path, classifier_path, dim)
SD.segment_dataset(tissue_dataset_border_path, seg_dataset_border_path, classifier_path, dim)

# %%
# 🧩 Tiled recombination

path_compartments = root_path + "compartments" 
seg_dataset_path = root_path + "inner_tiles/Neoplastic/"  
seg_dataset_border_path = root_path + "border_tiles/Neoplastic/" 

#path_mask = root_path + "extraction_mask/extraction_mask_CD34.png"
#path_mask = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient2/extraction_mask/extraction_mask_" + "_CD34.png"
path_mask = root_path + "extraction_mask/extraction_mask_FAP.png"
extraction_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
extraction_mask = 255 * extraction_mask

tile_size = 2048
down_factor = 50 

WSI_tissue_mask, WSI_tissue_smooth_mask, WSI_tissue_cleaned_mask = RCB.tiles_recombination(extraction_mask, seg_dataset_path, seg_dataset_border_path, down_factor, tile_size, correct_colors)

plt.imshow(WSI_tissue_mask)
plt.title('WSI_tissues_mask')
plt.axis('off')
plt.show()
plt.imsave(path_compartments + "/WSI_neoplastic_mask.png", WSI_tissue_mask, dpi=300)

plt.imshow(WSI_tissue_smooth_mask)
plt.title('WSI_tissue_smooth_mask')
plt.axis('off')
plt.show()    
plt.imsave(path_compartments + "/WSI_neoplastic_smoothed_mask.png", WSI_tissue_smooth_mask, dpi=300)

plt.imshow(WSI_tissue_cleaned_mask)
plt.title('WSI_tissue_cleaned_mask')
plt.axis('off')
plt.show()    
plt.imsave(path_compartments + "/WSI_neoplastic_cleaned_mask.png", WSI_tissue_cleaned_mask, dpi=300)


# %%
# 🗿 Performance metrics 

#path_semanticGT = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_semantic"
#path_semanticPRED = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_test"
path_semanticGT = root_path + "Semantic tiles"  
path_semanticPRED = root_path + "Predicted tiles 1"


ACC, PRE, REC = CM.compute_metrics(path_semanticGT, path_semanticPRED, dim)

[glob_acc, back_acc, neo_acc, nneo_acc] = ACC
[glob_pre, back_pre, neo_pre, nneo_pre] = PRE
[back_rec, neo_rec, nneo_rec] = REC

print(f"Accuracy : {glob_acc}")
print(f"Precision : {glob_pre}")

print(f"Accuracy background : {back_acc}")
print(f"Accuracy neoplastic : {neo_acc}")
print(f"Accuracy non neoplastic : {nneo_acc}")

print(f"Precision background : {back_pre}")
print(f"Precision neoplastic : {neo_pre}")
print(f"Precision non neoplastic : {nneo_pre}")

print(f"Recall background : {back_rec}")
print(f"Recall neoplastic : {neo_rec}")
print(f"Recall non neoplastic : {nneo_rec}")
