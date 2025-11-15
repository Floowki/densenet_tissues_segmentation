import numpy as np 
import cv2
from tqdm import tqdm 
import os
import torch

from architecture import CustomDenseNet


# Function for segmenting the dataset with the trained classifier 
def segment_dataset(dataset_path, seg_dataset_path, trained_classifier_path, dim) : 
    #| Args : 
    #|   # dataset_path : path to the folder containing the stain normalized tiles to segment              
    #|   # seg_dataset_path : path to the folder where to store the segmented tiles   
    #|   # trained_classifier_path : path to collect the classifier previously trained
    #|   # dim : dimension to divide the tile into (depends on the trained classifier)

    #| Outputs : 
    #|   # the tiles are segmented and stored in a given folder 
    
    
    # Load the trained classifier 
    checkpoint = torch.load(trained_classifier_path)          # Load the trained weights
    model_state_dict = checkpoint['model_state_dict']
    model = CustomDenseNet(output_size=dim)                                  # Backbone of customDenseNet 
    model.load_state_dict(model_state_dict, strict=False)     # load the usefull things for predictions 
    model.eval()                                              # Set the model to evaluation mode 
    
    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loop over the tiles in the dataset 
    filenames = os.listdir(dataset_path)
    counter = 0 

    for filename in tqdm(filenames, desc="Tissues segmentation", unit="tile", total=len(filenames)):
        
        counter = counter + 1 
        path_direction = os.path.join(seg_dataset_path, filename)
        
        if os.path.exists(path_direction):
            continue  # Skip and continue 
        
        
        # Load the current tile to process 
        img_source_path = os.path.join(dataset_path, filename)
        img_source = cv2.imread(img_source_path) 
        img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
        
        # Downsampled the image from 2048 x 2048 to 512 x 512
        if img_source.shape[0] != dim : 
            downsampled_image = cv2.resize(img_source, (dim, dim), interpolation=cv2.INTER_AREA)
        
        else : 
            downsampled_image = img_source
        
        # Check for applying / not applying the classifier : to discard wrong-normalised tiles
        mask_purple = cv2.inRange(downsampled_image, (67, 27, 64), (67, 27, 64))
        mask_black = cv2.inRange(downsampled_image, (0, 0, 0), (0, 0, 0))
        
        purple_ratio = np.mean(mask_purple/255)
        
        black_ratio = np.mean(mask_black/255)
        
        # If more than 10% of pixels are this purple, return a mask with only 0 (background)
        if (purple_ratio > 0.04) or (black_ratio > 0.03):
        
            img_segmented = np.zeros(downsampled_image.shape[:2], dtype=np.uint8)
            img_segmented_path = os.path.join(seg_dataset_path, filename)
            cv2.imwrite(img_segmented_path, img_segmented)
            
            continue
            
        
        image_tensor = torch.from_numpy(downsampled_image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.unsqueeze(0)
   
        # >>>> Use the trained classifier 
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
    
        # >>>> Convert the tensor to a numpy array
        img_segmented = predicted.cpu().squeeze().numpy()
        
        # Store the tile semantic mask (with 0, 1 and 2 as tissue labels)
        img_segmented_path = os.path.join(seg_dataset_path, filename)
        cv2.imwrite(img_segmented_path, img_segmented)
    
    return 
