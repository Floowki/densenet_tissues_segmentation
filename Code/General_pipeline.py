import cv2 

# 🎨 Manual tiles annotation

dim = 512                                                         # dimension of the images 
source_path = "C:/Users/augus/Desktop/Source tiles"               # path to source images 
semantic_path = "C:/Users/augus/Desktop/additional semantic"      # path to manual annotation masks 
semantic_desti = ""                                               # path for cleaned masks 

# 🧹 Data refinement 

#####   define a subfunction to perform the cleaning operation 

# 🗃️ PyTorch dataset integration

source_norm_desti = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_source"
semantic_desti = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/tiles_semantic"
 
df_DN = NNN.dataset_df(source_norm_desti, semantic_desti)

path_df_seg = "C:/Users/augus/Desktop/OSR_DATA_PROCESS/Pixel_classifier/"
df_DN.to_pickle(path_df_seg + "dataset_semantic_segmentation_512")

# 🚀 Data loader creation

# >> Split the dataset 
train_loader, val_loader = NNN.init_dataloader(df_DN, batch_size = 8, shuffle = True)

# 🏗️ Neural architecture design

# ⛳ Model fine-tuning


patience = 10
num_epochs = 20
epochs_no_improve = 0
metrics_name = 'training_metrics.csv' # saved in the current folder in Spyder 

NNN.model_FineTune(patience, num_epochs, epochs_no_improve, train_loader, val_loader, metrics_name)

# 👁️ Segmentation visualization 

def visualize_seg(test_loader) : 
    #| Args :              
    #|   # test_loader : dataloader for dealing with test batches         

    #| Outputs : 
    #|   # display the source / segmented images 
     
    model = CustomDenseNet()

    checkpoint = torch.load('C:/Users/augus/Desktop/Code/fine_tuned_densenet_norm_HE.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Select an example image from the test dataset
    start_time = time.time()
    example_image, example_mask = next(iter(test_loader))

    # Move the example image to the appropriate device 
    example_image = example_image.to(device)

    # Make a prediction on the example image 
    with torch.no_grad():
        output = model(example_image)
        _, predicted = torch.max(output, 1)


    # Convert the tensors to numpy arrays
    example_image_np = example_image.cpu().squeeze().permute(1, 2, 0).numpy()
    predicted_np = predicted.cpu().squeeze().numpy()
    end_time = time.time()
    
    rgb_semantic = np.zeros((predicted_np.shape[0], predicted_np.shape[1], 3), dtype="uint8")

    rgb_semantic[predicted_np == 0] = correct_colors[0]    # background
    rgb_semantic[predicted_np == 1] = correct_colors[1]    # stroma
    rgb_semantic[predicted_np == 2] = correct_colors[2]    # cellular

    # Plot the source image and the resulting semantic segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Source image
    axes[0].imshow(example_image_np)
    axes[0].set_title('Source Image')
    axes[0].axis('off')

    # Resulting semantic segmentation
    axes[1].imshow(rgb_semantic)
    axes[1].set_title('Semantic Segmentation')
    axes[1].axis('off')

    plt.show() 
    print(f"Segmentation time : {(end_time - start_time)}")
    
    return 


# 🌎 Full-scale dataset segmentation

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

# 🧩 Tiled recombination

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








