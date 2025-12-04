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

dataset_accuracy = 0
dataset_precision = 0
dataset_accuracy_back = 0
dataset_accuracy_neo = 0
dataset_accuracy_non_neo = 0
dataset_precision_back = 0
dataset_precision_neo = 0
dataset_precision_non_neo = 0

# Global precision tracking
TP_global = 0
P_global = 0  # Total predicted positives (non-background)

# Per-class accuracy tracking
TP_back = 0; TN_back = 0; FP_back = 0; FN_back = 0
TP_neo = 0;   TN_neo = 0;   FP_neo = 0;   FN_neo = 0
TP_non_neo = 0; TN_non_neo = 0; FP_non_neo = 0; FN_non_neo = 0

all_back = 0; all_neo = 0; all_non_neo = 0

filenames = os.listdir(path_semanticPRED)
for filename in tqdm(filenames, desc="Manually segmented masks loop", unit="tile"):
    maskGT_path = os.path.join(path_semanticGT, filename)
    maskPRED_path = os.path.join(path_semanticPRED, filename)

    mask_GT = cv2.imread(maskGT_path, cv2.IMREAD_GRAYSCALE)
    mask_PRED = cv2.imread(maskPRED_path, cv2.IMREAD_GRAYSCALE)

    for i in range(512):
        for j in range(512):
            gt = mask_GT[i, j]
            pred = mask_PRED[i, j]
            diff = abs(gt - pred)

            # Global accuracy
            if diff == 0:
                dataset_accuracy += 1

            # Global precision (non-background)
            if pred != 0:  # Predicted as neoplastic or non-neoplastic
                P_global += 1
                if gt == pred:  # Correct prediction
                    TP_global += 1

            # Background (class 0)
            if gt == 0 and pred == 0:
                TP_back += 1
                dataset_precision_back += 1
            elif gt == 0 and pred != 0:
                FP_back += 1
            elif gt != 0 and pred == 0:
                FN_back += 1
            else:
                TN_back += 1  # Non-background correctly not predicted as background
            all_back += 1 if pred == 0 else 0

            # Neoplastic (class 2)
            if gt == 2 and pred == 2:
                TP_neo += 1
                dataset_precision_neo += 1
            elif gt != 2 and pred == 2:
                FP_neo += 1
            elif gt == 2 and pred != 2:
                FN_neo += 1
            else:
                TN_neo += 1  # Non-neoplastic correctly not predicted as neoplastic
            all_neo += 1 if pred == 2 else 0

            # Non-neoplastic (class 1)
            if gt == 1 and pred == 1:
                TP_non_neo += 1
                dataset_precision_non_neo += 1
            elif gt != 1 and pred == 1:
                FP_non_neo += 1
            elif gt == 1 and pred != 1:
                FN_non_neo += 1
            else:
                TN_non_neo += 1  # Non-neoplastic correctly not predicted as non-neoplastic
            all_non_neo += 1 if pred == 1 else 0

# Normalize metrics
total_pixels = len(filenames) * 512 * 512
dataset_accuracy /= total_pixels

# Global precision
dataset_precision = TP_global / P_global if P_global > 0 else 0

# Per-class accuracy
dataset_accuracy_back = (TP_back + TN_back) / (TP_back + TN_back + FP_back + FN_back) if (TP_back + TN_back + FP_back + FN_back) > 0 else 0
dataset_accuracy_neo = (TP_neo + TN_neo) / (TP_neo + TN_neo + FP_neo + FN_neo) if (TP_neo + TN_neo + FP_neo + FN_neo) > 0 else 0
dataset_accuracy_non_neo = (TP_non_neo + TN_non_neo) / (TP_non_neo + TN_non_neo + FP_non_neo + FN_non_neo) if (TP_non_neo + TN_non_neo + FP_non_neo + FN_non_neo) > 0 else 0

# Per-class precision
dataset_precision_back /= all_back if all_back > 0 else 1
dataset_precision_neo /= all_neo if all_neo > 0 else 1
dataset_precision_non_neo /= all_non_neo if all_non_neo > 0 else 1

# Per-class recall 
recall_back = TP_back / (TP_back + FN_back) if (TP_back + FN_back) > 0 else 0
recall_neo = TP_neo / (TP_neo + FN_neo) if (TP_neo + FN_neo) > 0 else 0
recall_non_neo = TP_non_neo / (TP_non_neo + FN_non_neo) if (TP_non_neo + FN_non_neo) > 0 else 0


print(f"Accuracy : {dataset_accuracy}")
print(f"Precision : {dataset_precision}")

print(f"Accuracy background : {dataset_accuracy_back}")
print(f"Accuracy neoplastic : {dataset_accuracy_neo}")
print(f"Accuracy non neoplastic : {dataset_accuracy_non_neo}")

print(f"Precision background : {dataset_precision_back}")
print(f"Precision neoplastic : {dataset_precision_neo}")
print(f"Precision non neoplastic : {dataset_precision_non_neo}")

print(f"Recall background : {recall_back}")
print(f"Recall neoplastic : {recall_neo}")
print(f"Recall non neoplastic : {recall_non_neo}")


