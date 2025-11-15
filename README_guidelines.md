# 🎨 Manual tiles annotation

Our approach was conducted on ccRCC Whole-Slide Images from the TCGA database, a repository and computational platform for cancer researchers with thousands of kidney cancer histopathological images. \
🔗 https://portal.gdc.cancer.gov/analysis_page?app= \
A collection of 2048x2048 tiles was manually annotated and reviewed by an expert pathologist, distinguishing pixels into three categories: neoplastic (blue), non neoplastic (red) or background (black). This operation was performed using the application Sketchbook. 

<img src='Figures/Pipeline 1.jpg' width='100%'> 

```python
import cv2
import matplotlib.pyplot as plt 

import clean_dataset as CD
import construct_dataset as ConsD
import fine_tune_model as FTM
import test as TS
import segment_dataset as SD
import recombination as RCB

dim = 512   # dimension of the images    
root_path = "C:/Users/augus/Desktop/PORTFOLIO GITHUB/Dataset DenseNet fine tuning/"                      
source_path = root_path + "Source tiles"                                       # path to source images 
source_resized_path = root_path + "Source resized tiles"                       # path to save resized tiles
mask_path = root_path + "Mask tiles"                                           # path to manual annotation masks 
semantic_desti = root_path + "Semantic tiles"                                  # path for cleaned masks 
path_df_seg = root_path + "Pixel classifier/"                                  # path to save the dataset as a dataframe
classifier_path = root_path + "Pixel_classifier/fine_tuned_densenet_HE.pth"    # path to save the classifier once trained 
```

# 🧹 Data refinement 

The semantic masks delineating compartments cannot be used as direct inputs as they comprise 3 channels and may represent colors with close but distinct RGB tuple (R, G, B) in [0, 255]. The colour assigned to each compartment should be converted into semantic labels (0: background; 1: non neoplastic; 2: neoplastic). To that end, pixels channels are summed and associated with a label, while pixels presenting an unreferenced sum are assigned with the closest color for consistency. 

```python
correct_colors = {
    1: [255, 1, 13], # non neoplastic
    0: [1, 1, 1],    # background
    2: [1, 90, 255]  # neoplastic 
}

CD.construct_dataset(source_path, source_resized_path, mask_path, semantic_desti, dim, correct_colors)
```

# 🗃️ PyTorch dataset integration

The dataset containing images and semantic masks might be to heavy to be loaded and manipulated all at once. Deep learning workflows manipulation with PyTorch resort to PyTorch datasets for accessing data in a predictable format and loading data in batches for efficient training process and memory efficiency. A dataframe compiles the dataset characteristics and links to the images for proper loading when necessary. 

```python
df_DN = ConsD.dataset_df(source_resized_path, semantic_desti)
df_DN.to_pickle(path_df_seg + "dataset_semantic_segmentation")
```

# 🚀 Data loader creation

Then Dataloader objects work hand-in-hand with a Dataset to streamline the process of feeding data into the model during training and evaluation. The split is done stratifying on the basis of the dominant class on images. 

```python
train_loader, val_loader = ConsD.init_dataloader(df_DN, batch_size = 8, shuffle = True)
```
 
# 🏗️ Neural architecture design

The pre-trained model weights can be collected directtly from a PyTorch package, associated with a custom segmentation head and optimized for the specific segmentation task. 

```python
from torchvision import models
...
self.densenet = models.densenet169(pretrained=True)
...
```

<img src='Figures/Pipeline 2.jpg' width='100%'> 
 
# ⛳ Model fine-tuning

The custom DenseNet architecture is retrained upon the dataset for 20 epochs. The Loss function and the accuracy are monitored during the process. 

```python
patience = 10
num_epochs = 20
metrics_name = 'training_metrics.csv' # saved in the current folder 

FTM.model_FineTune(patience, num_epochs, train_loader, val_loader, dim, metrics_name)
```

<img src='Figures/Pipeline 3.jpg' width='100%'> 
 
# 👁️ Segmentation validation & visualization

```python
img_path = ""

img = cv2.imread(img_path) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Image source")
plt.axis("off")
plt.show()
```

<img src='Figures/Pipeline 4.jpg' width='100%'> 
 
# 🌎 Full-scale dataset segmentation

```python

```
 
# 🧩 Tiled recombination

```python

```

<img src='Figures/Pipeline 5.jpg' width='100%'> 
 
