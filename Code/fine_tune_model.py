import numpy as np 
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict





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