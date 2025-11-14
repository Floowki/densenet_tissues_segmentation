# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 13:57:35 2025
@author: Flowki

Code for peforming cellular / stroma segmentation 
"""


import torch  
from torchvision import models  
from tqdm import tqdm 
import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import copy 
import torch.nn.functional as F 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time 
import re 



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


def dataset_df(Source_path, SemMask_path, ) : 
    #| Inputs : 
    #|   # Source_path : path to the source tiles 
    #|   # SemMask_path : path to the manual semantic mask 

    #| Outputs : 
    #|   # df_DN : dataframe of the dataset 
    
    df_DN = pd.DataFrame(columns=['Tile', 'Semantic mask'])

    filenames = os.listdir(SemMask_path)
 
    for idx, filename in enumerate(filenames) : 
        
        img_path = os.path.join(Source_path, filename)
        mask_path = os.path.join(SemMask_path, filename)
        
        df_DN.loc[len(df_DN)] = [img_path, mask_path]
          

    # Get the class distribution for balance / imbalance knowledge 
    df_DN["Class distribution"] = df_DN["Semantic mask"].apply(get_class_distrib) # add the column to the dataframe 

    # Stratify on dominant class ground (background, cellular, or stroma)
    df_DN["Dominant class"] = df_DN["Class distribution"].apply(get_dominant_class) # add the column to the dataframe 

    
    train_df_DN, val_df_DN = train_test_split(
        df_DN, 
        test_size=0.3, 
        stratify=df_DN["Dominant class"],  # Ensures class balance
        random_state=42
    )

    train_df_DN["Split"] = "train"
    val_df_DN["Split"] = "val"
    df_DN = pd.concat([train_df_DN, val_df_DN])
    
    return df_DN


# Blue  -> Class 1 CELLULAR : sum of channels of cleaned semantic mask 88
# Red   -> Class 2 STROMA :  sum of channels of cleaned semantic mask 12
# Black -> Class 0 BACKGROUND :  sum of channels of cleaned semantic mask 3 
    


def get_class_distrib(semantic_mask_path) :
    #| Inputs : 
    #|   # semantic_mask_path : path to the GT semantic masks for training

    #| Outputs : 
    #|   # distrib : class distribution of the input label mask 
    
    #semantic_mask = cv2.imread(semantic_mask_path)
    semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)

    height, width = semantic_mask.shape
    total = height * width
    
    distrib = {
        "background": np.sum(semantic_mask == 0) / total,
        "cellular": np.sum(semantic_mask == 2) / total,
        "stroma": np.sum(semantic_mask == 1) / total
    }
    
    return distrib 

def get_dominant_class(distribution): 
    #| Inputs : 
    #|   # distribution : the distribution of classes of the form {'background': 0.29, 'cellular': 0.53, 'stroma': 0.16}

    #| Outputs : 
    #|   # dominant_class : the dominant class wrt percentage 
    
    # Define the distribution

    dominant_class = max(distribution, key=distribution.get)

    return dominant_class 


# Create a PyTorch dataset (efficient manipulation)
class WSIDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_path = row["Tile"]
        mask_path = row["Semantic mask"]

        # Load images with error handling
        try:
            # Load patch as RGB
            patch = cv2.imread(patch_path)
            if patch is None:
                raise ValueError(f"Failed to load image at {patch_path}")
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # Load mask as grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask at {mask_path}")

            # Convert to PyTorch tensors and normalize
            patch = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0  # (3, 512, 512), [0, 1]
            mask = torch.from_numpy(mask).long()  # (512, 512)

            return patch, mask

        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            # Return a dummy tensor if there's an error (you might want to handle this differently)
            return torch.zeros((3, 512, 512)), torch.zeros((512, 512), dtype=torch.long)

def init_dataloader(df_DN, batch_size, shuffle) : 
    #| Inputs : 
    #|   # df_DN : dataframe for constructing the dataset 
    #|   # batch_size : the number of images in a batch 
    #|   # shuffle : boolean for shuffling or not 

    #| Outputs : 
    #|   # train_loader : loader for training data from path in the dataframe 
    #|   # val_loader : loader for validation data from path in the dataframe 
        
    train_df_DN = df_DN[df_DN["Split"] == "train"]
    val_df_DN = df_DN[df_DN["Split"] == "val"]

    # Create dataloaders WITHOUT augmentation
    train_loader = DataLoader(WSIDataset(train_df_DN), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WSIDataset(val_df_DN), batch_size=batch_size)
    
    return train_loader, val_loader
    

#### DENSENET 169 ####
class CustomDenseNet(nn.Module):
    def __init__(self):
        super(CustomDenseNet, self).__init__()
        # Load pre-trained DenseNet169
        self.densenet = models.densenet169(pretrained=True)

        # Remove original classifier
        self.densenet.classifier = nn.Identity()

        # Add a segmentation head that maintains spatial dimensions
        self.upsample = nn.Sequential(
            # First upsampling block
            nn.Conv2d(1664, 1024, kernel_size=3, padding=1),  # DenseNet169 has 1664 features
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Second upsampling block
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Third upsampling block
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Final convolution to get to num_classes
            nn.Conv2d(256, 3, kernel_size=1)
        )

    def forward(self, x):
        
        # Extract features from the backbone
        features = self.densenet.features(x)

        # Apply the segmentation head
        output = self.upsample(features)

        # Ensure output has the correct spatial dimensions
        if output.shape[2] != 512 or output.shape[3] != 512:
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=True)

        return output



def model_FineTune(patience, num_epochs, epochs_no_improve, train_loader, val_loader, metrics_name) : 
    #| Inputs : 
    #|   # patience : patience level 
    #|   # num_epochs : total number of epochs
    #|   # epochs_no_improve : the number of allowed successive epochs without improvement 
    #|   # train_loader : the loader getting the training images 
    #|   # val_loader : the loader getting the validation images 
    #|   # metrics_name : name of the file to save the training metrics 

    #| Outputs : 
    #|   # metrics_log : info on training saved on a CSV  
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDenseNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    
    # Apply class weights to balance the dataset 
    class_weights = torch.tensor([0.45, 0.24, 0.31], dtype=torch.float)
    class_weights = class_weights.to(device) 
    
    best_loss = float('inf')
    metrics_log = []

    for epoch in tqdm(range(num_epochs), desc="Segmentation model training", unit="epoch", total=num_epochs):
        
        # Training
        train_loss, train_acc, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validation
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        
        # Log metrics
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        metrics_log.append(epoch_log)
        
        # Print summary
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
    # Save metrics to CSV
    pd.DataFrame(metrics_log).to_csv(metrics_name, index=False)
    
    df_metrics = pd.read_csv(metrics_name)

    plt.figure(figsize=(12, 4))
    plt.plot(df_metrics['epoch'], df_metrics['train_loss'], label='Train')
    plt.plot(df_metrics['epoch'], df_metrics['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure(figsize=(12, 4))
    plt.plot(df_metrics['epoch'], df_metrics['train_accuracy'], label='Train')
    plt.plot(df_metrics['epoch'], df_metrics['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Save the classifier 
    FTnetwork_name = 'fine_tuned_densenet_norm_HE.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': best_loss,
    }, FTnetwork_name)  # Save as .pth or .pt file
    
    return 


def compute_accuracy(output, target):
    #| Inputs : 
    #|   # output : segmentation result of the classifier                 
    #|   # target : groundtruth segmentation semantic mask            

    #| Outputs : 
    #|   # accuracy : accuracy of the model segmentation (metric)
    
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)  # Get predicted class (0,1,2)
        correct = (pred == target).float().sum()
        accuracy = correct / target.numel()  # % correct pixels
        
    accuracy = accuracy.item()
        
    return accuracy


def train_one_epoch(model, train_loader, optimizer, criterion, device) :
    #| Inputs : 
    #|   # model : the model architecture to load                
    #|   # train_loader : the train loader collecting the images for training
    #|   # optimizer : optimizer used for training 
    #|   # criterion : the criterion used for the Loss function minimization  
    #|   # device : the device the training is delegated to        

    #| Outputs : 
    #|   # epoch_loss : current epoch loss  
    #|   # epoch_acc : current accuracy metric 
    
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_pixels = 0

    for inputs, masks in train_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        total = masks.numel()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == masks.data)
        total_pixels += total  

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (running_corrects.double()).item() / total_pixels
    

    return epoch_loss, epoch_acc, {}



def compute_class_metrics(outputs, masks):
    #| Args : 
    #|   # outputs : the outputs of the model for classification            
    #|   # masks : the GT masks used for comparing with the model outputs      

    #| Outputs : 
    #|   # metrics : dictionary of class-wise metrics
    
  
    _, predicted = torch.max(outputs, 1)
    metrics = {}

    # Compute metrics for each class
    for class_id in range(3):  # Assuming 3 classes
        true_positives = ((predicted == class_id) & (masks == class_id)).sum().item()
        false_positives = ((predicted == class_id) & (masks != class_id)).sum().item()
        false_negatives = ((predicted != class_id) & (masks == class_id)).sum().item()

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        metrics[f'precision_class_{class_id}'] = precision
        metrics[f'recall_class_{class_id}'] = recall
        metrics[f'f1_score_class_{class_id}'] = f1_score

    return metrics


def validate(model, dataloader, criterion, device):
    #| Inputs : 
    #|   # model : the model archutecture defined previously              
    #|   # dataloader : dataloader for dealing with batches         
    #|   # criterion : criterion used for loss function (here cross-entropy minimization)    
    #|   # device : GPU if available else CPU  

    #| Outputs : 
    #|   # val_loss : the loss for the validation batch 
    #|   # val_accuracy : the accuracy of the validation batch 
    #|   # val_class_metrics : the performance metrics for the validation batch 
    
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_class_metrics = defaultdict(float)
    
    with torch.no_grad():
        for patches, masks in dataloader:
            patches, masks = patches.to(device), masks.to(device)
            outputs = model(patches)
            
            val_loss += criterion(outputs, masks).item()
            val_accuracy += compute_accuracy(outputs, masks)
            
            metrics = compute_class_metrics(outputs, masks)
            for key, val in metrics.items():
                val_class_metrics[key] += val
    
    val_loss /= len(dataloader)
    val_accuracy /= len(dataloader)
    for key in val_class_metrics:
        val_class_metrics[key] /= len(dataloader)
    
    return val_loss, val_accuracy, val_class_metrics



def segmentation_tests(test_loader) : 
    #| Args :              
    #|   # test_loader : dataloader for dealing with test batches         

    #| Outputs : 
    #|   # display some metrics of the tests 
    
    model = CustomDenseNet()

    # Load the state dictionary
    checkpoint = torch.load('C:/Users/augus/Desktop/Code/fine_tuned_densenet_norm_HE.pth')

    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Make predictions on the test dataset
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(masks.cpu().numpy())

    actuals_flat = [item for sublist in actuals for item in sublist.flatten()]
    predictions_flat = [item for sublist in predictions for item in sublist.flatten()]

    accuracy = accuracy_score(actuals_flat, predictions_flat)
    precision = precision_score(actuals_flat, predictions_flat, average='macro')
    recall = recall_score(actuals_flat, predictions_flat, average='macro')
    f1 = f1_score(actuals_flat, predictions_flat, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return 



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


# %% 

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
    model = CustomDenseNet()                                  # Backbone of customDenseNet 
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


# %% 

correct_colors = {
    1: [255, 1, 13], # stroma
    0: [1, 1, 1],    # background
    2: [1, 90, 255]  # cellular
}


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



def tiles_recombination(extraction_mask, semantic_masks_path, semantic_masks_border_path, down_factor, tile_size) : 
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


 
