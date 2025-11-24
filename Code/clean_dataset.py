import numpy as np
import os
from tqdm import tqdm 
import cv2 



def construct_dataset(source_path, source_resized_path, semantic_path, semantic_desti, dim, correct_colors) : 
#| Inputs : 
#|   # source_path : path to the 2048x2048 sources tiles 
#|   # source_resizedd_path : path to save the resized source tiles 
#|   # semantic_path : path to the 2048x2048 semantic masks 
#|   # semantic_desti : path where to save the resize instance masks 
#|   # dim : dimension the images should be tiles into (e.g 2048 --> 512)  

#| Outputs : 
#|   # subtiles_source : the subtiles from the source tiles with correct dimensions 
#|   # subtiles_semantic : the subtiles fronm the semantic masks with correct dimensions 

## Steps ## 

    # >> Loop over tiles     
    filenames = os.listdir(source_path)

    for filename in tqdm(filenames, desc="Semantic conversion", unit="tile"):
        
        img_path = os.path.join(source_path, filename)
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(semantic_path, filename)
        mask = cv2.imread(mask_path) 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        
        # >> Check source image dimension
        if img.shape[0] != dim :
            img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)

        # >> Check mask image dimension
        if mask.shape[0] != dim :
            mask = cv2.resize(mask, (dim, dim), interpolation=cv2.INTER_AREA)

        # >> Store resized source tiles (even if not)
        img_resized_path = os.path.join(source_resized_path, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_resized_path, img)

        # correct_colors definition 
        # sum axis=-1 
        # red : 13
        # black : 3
        # blue : 90 
        
        def nearest_color(pixel):
            # Calculate the Euclidean distance between the pixel and each correct color
            distances = {color: np.linalg.norm(np.array(pixel) - np.array(rgb)) for color, rgb in correct_colors.items()}
            # Return the color with the smallest distance
            return min(distances, key=distances.get)
        
        
        for i in range(dim):
            for j in range(dim):
                pixel = tuple(mask[i, j])
                # Replace the pixel with the nearest correct color if it's not already correct
                if pixel not in correct_colors.values():
                    nearest = nearest_color(pixel)
                    mask[i, j] = correct_colors[nearest]
        
    
        semantic_mask = np.sum(mask, axis=-1).astype("uint8")
        semantic_mask[semantic_mask == 3] = 0      # background
        semantic_mask[semantic_mask == 13] = 1     # stroma 
        semantic_mask[semantic_mask == 90] = 2     # cellular 
        
        # >> Save the image in the folder 
        mask_name = os.path.join(semantic_desti, filename)
        cv2.imwrite(mask_name, semantic_mask)
                
    return 