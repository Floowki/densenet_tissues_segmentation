from torch.utils.data import DataLoader





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