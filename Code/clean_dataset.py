import numpy as np
import os
from tqdm import tqdm 
import cv2 


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






    def construct_dataset(source_path, semantic_path, source_desti, semantic_desti, dim) : 
    #| Inputs : 
    #|   # source_path : path to the 2048x2048 sources tiles 
    #|   # semantic_path : path to the 2048x2048 semantic masks 
    #|   # source_desti : path where to save the resize source tiles 
    #|   # semantic_desti : path where to save the resize instance masks 
    #|   # dim : dimension the images should be tiles into (e.g 2048 --> 512)  

    #| Outputs : 
    #|   # subtiles_source : the subtiles from the source tiles with correct dimensions 
    #|   # subtiles_semantic : the subtiles fronm the semantic masks with correct dimensions 
    
    ## Steps ## 
        nb_quad = 2048 // dim 
        
        filenames = os.listdir(source_path)

        for filename in tqdm(filenames, desc="Subtiles extraction", unit="tile"):
            
            img_path = os.path.join(source_path, filename)
            img = cv2.imread(img_path) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask_path = os.path.join(semantic_path, filename)
            mask = cv2.imread(mask_path) 
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        
        
            correct_colors = {
                'red': [255, 1, 13],
                'black': [1, 1, 1],
                'blue': [1, 90, 255]
            }
            # sum axis=-1 
            # red : 13
            # black : 3
            # blue : 90 
            
            def nearest_color(pixel):
                # Calculate the Euclidean distance between the pixel and each correct color
                distances = {color: np.linalg.norm(np.array(pixel) - np.array(rgb)) for color, rgb in correct_colors.items()}
                # Return the color with the smallest distance
                return min(distances, key=distances.get)
            
            
            for i in range(2048):
                for j in range(2048):
                    pixel = tuple(mask[i, j])
                    # Replace the pixel with the nearest correct color if it's not already correct
                    if pixel not in correct_colors.values():
                        nearest = nearest_color(pixel)
                        mask[i, j] = correct_colors[nearest]
            
        
            semantic_mask = np.sum(mask, axis=-1).astype("uint8")
            semantic_mask[semantic_mask == 3] = 0      # background
            semantic_mask[semantic_mask == 13] = 1              # stroma 
            semantic_mask[semantic_mask == 90] = 2     # cellular 
            
            for i in range(nb_quad) :      # row loop 
                for j in range(nb_quad) :  # col loop 
                
                    #quadrant_name = filename + "_" + str(4*i+j+1) + ".png" # from to the nb_quad 
                    base_name, ext = os.path.splitext(filename)
                    suffix = str(nb_quad*i+j+1)
                    quadrant_name = quadrant_name = f"{base_name}_{suffix}{ext}"
                    
                    img_quadrant_path = os.path.join(source_desti, quadrant_name)
                    img_quadrant = img[i*dim: (i+1)*dim, j*dim: (j+1)*dim, :]
                    cv2.imwrite(img_quadrant_path, cv2.cvtColor(img_quadrant, cv2.COLOR_RGB2BGR))
                    
                    mask_quadrant_path = os.path.join(semantic_desti, quadrant_name) 
                    mask_quadrant = semantic_mask[i*dim: (i+1)*dim, j*dim: (j+1)*dim]
                    cv2.imwrite(mask_quadrant_path, mask_quadrant)
                    
        return 