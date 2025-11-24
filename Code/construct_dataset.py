import numpy as np 
import pandas as pd 
import os
import cv2 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


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

    wgh_back, wgh_neo, wgh_non_neo = 0, 0, 0
    for i in range(len(train_df_DN)): 
    
        wgh_back = wgh_back + train_df_DN['Class distribution'].iloc[i]['background']
        wgh_neo = wgh_neo + train_df_DN['Class distribution'].iloc[i]['cellular']
        wgh_non_neo = wgh_non_neo + train_df_DN['Class distribution'].iloc[i]['stroma']

    wgh_back, wgh_neo, wgh_non_neo = wgh_back/len(train_df_DN), wgh_neo/len(train_df_DN), wgh_non_neo/len(train_df_DN) 
    wgh_back, wgh_neo, wgh_non_neo = 1/(3*wgh_back), 1/(3*wgh_neo), 1/(3*wgh_non_neo)
    sum_to_norm = wgh_back + wgh_neo + wgh_non_neo
    wgh_back, wgh_neo, wgh_non_neo = wgh_back/sum_to_norm, wgh_neo/sum_to_norm, wgh_non_neo/sum_to_norm

    weights_class = [wgh_back, wgh_neo, wgh_non_neo]
    
    return df_DN, weights_class


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
    train_loader = DataLoader(WSIDataset(train_df_DN), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(WSIDataset(val_df_DN), batch_size=batch_size)
    
    return train_loader, val_loader