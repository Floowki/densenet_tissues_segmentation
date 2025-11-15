import numpy as np 
import cv2 
import os
from tqdm import tqdm
import re
import copy




def downsample_spatial_dimensions_decimation(array, factor):
    #| Args : 
    #|   # array : a 3D array image (3-channel image) to downsample              
    #|   # factor : decimation downsampling factor    

    #| Outputs : 
    #|   # downsampled : the downsampled array 
    

    d_factor, h_factor = factor, factor

    # Select every nth element along each spatial dimension
    downsampled = array[::d_factor, ::h_factor, :]

    return downsampled



def tiles_recombination(extraction_mask, semantic_masks_path, semantic_masks_border_path, down_factor, tile_size, correct_colors) : 
    #| Args : 
    #|   # extraction_mask : final mask employed for biological tissue extraction 
    #|   # semantic_masks_path : path containing the semantic masks identifying 0, 1 and 2 (inner)
    #|   # semantic_masks_border_path : path containing the semantic masks identifying 0, 1 and 2 (border)
    #|   # down_factor : downsampling factor for extraction of the biological sample 
    #|   # tile_size : dimension of the tile in pixels  
        
    #| Outputs :
    #|   # WSI_tissue_mask : the colored global mask recombining the segmented tiles 
    #|   # WSI_semantic_mask_smoothed : the global semantic mask recombing the semantic tiles 
    
    
    # Initialize the global tissue map and global semantic mask
    H, W = extraction_mask.shape
    WSI_tissue_mask = np.zeros((H, W, 3), dtype="uint8") 
    WSI_semantic_mask = np.zeros((H, W), dtype="uint8")
       
    # Access the path and collect the inner masks
    filenames = os.listdir(semantic_masks_path)
    
    for index, filename in tqdm(enumerate(filenames), desc="Recombination kernel", unit="tile", total=len(filenames)):
    
        # Access the semantic segmentation masks     
        semantic_path = os.path.join(semantic_masks_path, filename)
        tile_semantic = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE) 
        
        # Collect the patch position in name 
        match = re.match(r'P(\d+)_T(\d+)_(\d+)_(\d+)', filename)
        if match : 
            row = int(match.group(2))
            row = row//down_factor 
            col = int(match.group(3))
            col = col//down_factor 
        else : 
            row, col = None, None 
    
        # Initialize the rgb semantic tile for visualization afterwards 
        tile_rgb_semantic = np.zeros((tile_semantic.shape[0], tile_semantic.shape[1], 3), dtype="uint8")

        tile_rgb_semantic[tile_semantic == 0] = correct_colors[0]    # background
        tile_rgb_semantic[tile_semantic == 1] = correct_colors[1]    # stroma
        tile_rgb_semantic[tile_semantic == 2] = correct_colors[2]    # cellular
        
        # Downsample 
        tile_rgb_semantic_down = cv2.resize(tile_rgb_semantic, (41, 41), interpolation=cv2.INTER_AREA)
        tile_semantic_down = cv2.resize(tile_semantic, (41, 41), interpolation=cv2.INTER_NEAREST)
        
        # Placing the semantic tile on the global semantic map 
        WSI_semantic_mask[row: row + tile_size//down_factor + 1, col: col + tile_size//down_factor + 1] = tile_semantic_down
        
        # Placing the tissues tile on the global tissue map 
        WSI_tissue_mask[row: row + tile_size//down_factor + 1, col: col + tile_size//down_factor + 1, :] = tile_rgb_semantic_down
    
    # Access the path and collect the border masks 
    filenames = os.listdir(semantic_masks_border_path)
    
    for index, filename in tqdm(enumerate(filenames), desc="Recombination kernel", unit="tile", total=len(filenames)):
    
        # Access the semantic segmentation masks     
        semantic_path = os.path.join(semantic_masks_border_path, filename)
        tile_semantic = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE) 
        
        # Collect the patch position in name 
        match = re.match(r'P(\d+)_T(\d+)_(\d+)_(\d+)', filename)
        if match :
            row = int(match.group(2))
            row = row//down_factor 
            col = int(match.group(3))
            col = col//down_factor 
        else : 
            row, col = None, None 
    
        # Initialize the rgb semantic tile for visualization afterwards 
        tile_rgb_semantic = np.zeros((tile_semantic.shape[0], tile_semantic.shape[1], 3), dtype="uint8")

        tile_rgb_semantic[tile_semantic == 0] = correct_colors[0]    # background
        tile_rgb_semantic[tile_semantic == 1] = correct_colors[1]    # stroma
        tile_rgb_semantic[tile_semantic == 2] = correct_colors[2]    # cellular
        
                    
        # Downsample 
        tile_rgb_semantic_down = cv2.resize(tile_rgb_semantic, (41, 41), interpolation=cv2.INTER_AREA)
        tile_semantic_down = cv2.resize(tile_semantic, (41, 41), interpolation=cv2.INTER_NEAREST)
        
        # Placing the semantic tile on the global semantic map 
        WSI_semantic_mask[row: row + tile_size//down_factor + 1, col: col + tile_size//down_factor + 1] = tile_semantic_down
        
        # Placing the tissues tile on the global tissue map 
        WSI_tissue_mask[row: row + tile_size//down_factor + 1, col: col + tile_size//down_factor + 1, :] = tile_rgb_semantic_down
    
    # Post processing for global semantic mask smoothing 
    # >>> Binary masks for each compartment
    #non_neoplastic_global = (WSI_semantic_mask == 1).astype(np.uint8)
    neoplastic_global = (WSI_semantic_mask == 2).astype(np.uint8)
    
    # >>> Morphological closing to each binary mask
    kernel = np.ones((5,5), np.uint8)
    
    # >>> Dilate the neoplastic region. erode the non neoplastic regions 
    dilated_neoplastic = cv2.dilate(neoplastic_global, kernel, iterations=2) 
    dilated_neoplastic = cv2.erode(dilated_neoplastic, kernel, iterations=1) 
    
    # >>> Combine the masks, giving precedence to the dilated neoplastic region
    smoothed_mask = copy.deepcopy(WSI_semantic_mask) #np.zeros_like(WSI_semantic_mask, dtype=np.uint8)
    smoothed_mask[dilated_neoplastic == 1] = 2

    WSI_semantic_mask_smoothed = np.zeros((H, W, 3), dtype="uint8") 
    
    WSI_semantic_mask_smoothed[smoothed_mask == 0] = correct_colors[0]    # background
    WSI_semantic_mask_smoothed[smoothed_mask == 1] = correct_colors[1]    # non neoplastic
    WSI_semantic_mask_smoothed[smoothed_mask == 2] = correct_colors[2]    # neoplastic
    
    
    return WSI_tissue_mask, WSI_semantic_mask_smoothed