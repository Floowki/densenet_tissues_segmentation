import cv2 

# 🎨 Manual tiles annotation

dim = 512                                                         # dimension of the images 
source_path = "C:/Users/augus/Desktop/Source tiles"               # path to source images 
semantic_path = "C:/Users/augus/Desktop/additional semantic"      # path to manual annotation masks 
semantic_desti = ""                                               # path for cleaned masks 

# 🧹 Data refinement 

# 🗃️ Data framework construction

# 🔗 PyTorch dataset integration

# 🚀 Data loader creation

# 🏗️ Neural architecture design

# ⛳ Model fine-tuning

# 👁️ Segmentation validation & visualization

# 🌎 Full-scale dataset segmentation

# 🧩 Tiled recombination









dim = 512 
source_path = "C:/Users/augus/Desktop/Source tiles"
semantic_path = "C:/Users/augus/Desktop/additional semantic"
semantic_desti = ""

# >> Prepare the dataset for training (manual groundtruth masks)
filenames = os.listdir(semantic_path)

for filename in tqdm(filenames, desc="Downsampling tiles", unit="tile", total=len(filenames)):
    
    # Source semantic mask 
    # >> Load the current semantic tile (GT mask) to process 
    mask_source_path = os.path.join(semantic_path, filename) 
    mask_source = cv2.imread(mask_source_path) 
    mask_source = cv2.cvtColor(mask_source, cv2.COLOR_BGR2RGB)
    
    # >> Preprocess the semantic mask to be sure to have only 3 classes 
    correct_colors = {
        'red': [255, 1, 13],
        'black': [1, 1, 1],
        'blue': [1, 90, 255]
    }  # sum axis=-1 : red=13, black=3, blue=90 
    
    def nearest_color(pixel):
        # Calculate the Euclidean distance between the pixel and each correct color
        distances = {color: np.linalg.norm(np.array(pixel) - np.array(rgb)) for color, rgb in correct_colors.items()}
        # Return the color with the smallest distance
        return min(distances, key=distances.get)
    
    
    for i in range(512):
        for j in range(512):
            pixel = tuple(mask_source[i, j])
            # Replace the pixel with the nearest correct color if it's not already correct
            if pixel not in correct_colors.values():
                nearest = nearest_color(pixel)
                mask_source[i, j] = correct_colors[nearest]
    

    mask_source = np.sum(mask_source, axis=-1).astype("uint8") 
    mask_source[mask_source == 3] = 0      # background
    mask_source[mask_source == 13] = 1     # stroma 
    mask_source[mask_source == 90] = 2     # cellular 
    
    # >> Save the image in the folder 
    mask_name = os.path.join(semantic_desti, filename)
    cv2.imwrite(mask_name, mask_source)

# >> Create and store the PyTorch dataframe 
source_norm_desti = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_source"
semantic_desti = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_semantic"
 
df_DN = NNN.dataset_df(source_norm_desti, semantic_desti)

path_df_seg = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/"
df_DN.to_pickle(path_df_seg + "dataset_semantic_segmentation_512")

# %%
# >> Split the dataset 
train_loader, val_loader = NNN.init_dataloader(df_DN, batch_size = 8, shuffle = True)
# >> Fine-tune the DenseNet network 
patience = 10
num_epochs = 20
epochs_no_improve = 0
metrics_name = 'training_metrics.csv' # saved in the current folder in Spyder 

NNN.model_FineTune(patience, num_epochs, epochs_no_improve, train_loader, val_loader, metrics_name)

# %%
# >> Compute performance metrics on a dataset 
path_semanticGT = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_semantic"
path_semanticPRED = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_test"

dataset_accuracy = 0
dataset_precision_back = 0
dataset_precision_neo = 0
dataset_precision_non_neo = 0
all_back = 0 
all_neo = 0 
all_non_neo = 0 

filenames = os.listdir(path_semanticPRED) 

for filename in tqdm(filenames, desc="Manually segmented masks loop", unit="tile") :

    maskGT_path = os.path.join(path_semanticGT, filename)
    maskPRED_path = os.path.join(path_semanticPRED, filename)
    
    # Collect GT mask 
    mask_GT = cv2.imread(maskGT_path, cv2.IMREAD_GRAYSCALE)
    
    # Collect PRED mask 
    mask_PRED = cv2.imread(maskPRED_path, cv2.IMREAD_GRAYSCALE)

    for i in range(512) :      
        for j in range(512) : 

            diff = abs(mask_GT[i, j] - mask_PRED[i, j]) 

            # Increment the accuracy 
            if diff == 0 : 
                
                dataset_accuracy = dataset_accuracy + 1 

            # Increment the background precision 
            if (mask_GT[i, j] == 0) and (mask_PRED[i, j] == 0) : 
                dataset_precision_back += 1
                all_back += 1
                
            if (mask_PRED[i, j] == 0) and (mask_GT[i, j] != 0) : 
                all_back += 1
            
            # Increment the neoplastic precision 
            if (mask_GT[i, j] == 2) and (mask_PRED[i, j] == 2) : 
                dataset_precision_neo += 1
                all_neo += 1
                
            if (mask_PRED[i, j] == 2) and (mask_GT[i, j] != 2) : 
                all_neo += 1
            
            # Increment the non neoplastic precision 
            if (mask_GT[i, j] == 1) and (mask_PRED[i, j] == 1) : 
                dataset_precision_non_neo += 1
                all_non_neo += 1
                
            if (mask_PRED[i, j] == 1) and (mask_GT[i, j] != 1) : 
                all_non_neo += 1

dataset_accuracy = dataset_accuracy / (len(filenames)*512*512)
dataset_precision_back = dataset_precision_back / all_back
dataset_precision_neo = dataset_precision_neo / all_neo
dataset_precision_non_neo = dataset_precision_non_neo / all_non_neo

print(f"Accuracy : {dataset_accuracy}")
print(f"Precision background : {dataset_precision_back}")
print(f"Precision neoplastic : {dataset_precision_neo}")
print(f"Precision non neoplastic : {dataset_precision_non_neo}")

# %%
# 🔰 Perform pathological segmentation

# Perform segmentation on dataset 
# >> paths inner/border tiles (source normalised)
tissue_dataset_path = root_path + "inner_tiles/HE_norm" 
tissue_dataset_border_path = root_path + "border_tiles/HE_norm" 

# >> destination inner/border tiles 
seg_dataset_path = root_path + "inner_tiles/Neoplastic" 
seg_dataset_border_path = root_path + "border_tiles/Neoplastic" 

# >> trained classifier path
classifier_path = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/fine_tuned_densenet_norm_HE.pth"
dim = 512 # dimension of the tiles used for training and inference then (512x512)

# Segment inner & border tiles 
NNN.segment_dataset(tissue_dataset_path, seg_dataset_path, classifier_path, dim)
NNN.segment_dataset(tissue_dataset_border_path, seg_dataset_border_path, classifier_path, dim)

# %%
# Conduct tiles recombination 
# >> paths 
# seg_dataset_path = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient1/inner_tiles/Neoplastic/A2"
# seg_dataset_border_path = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient1/border_tiles/Neoplastic/A2"
# path_compartments = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient1/compartments/"
seg_dataset_path = root_path + "inner_tiles/Neoplastic"  
seg_dataset_border_path = root_path + "border_tiles/Neoplastic"  
path_compartments = root_path + "compartments" 

# >> collect the extraction mask 
#path_mask = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Patient1/extraction_mask/extraction_mask_A2_CD34.png"
extraction_mask = cv2.imread(save_path_extraction, cv2.IMREAD_GRAYSCALE)
extraction_mask = 255 * extraction_mask

tile_size = 2048
down_factor = 50 

WSI_tissue_norm_mask, WSI_semantic_norm_mask = NNN.tiles_recombination(extraction_mask, seg_dataset_path, seg_dataset_border_path, down_factor, tile_size)

plt.imshow(WSI_tissue_norm_mask)
plt.title('WSI_tissues_norm_mask')
plt.axis('off')
plt.show()
plt.imsave(path_compartments + "WSI_neoplastic_mask" + ".png", WSI_tissue_norm_mask, dpi=300)

plt.imshow(WSI_semantic_norm_mask)
plt.title('WSI_semantic_norm_mask')
plt.axis('off')
plt.show()    
plt.imsave(path_compartments + "WSI_neoplastic_smoothed_mask" + ".png", WSI_semantic_norm_mask, dpi=300)