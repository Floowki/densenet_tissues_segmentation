# 🎯 Automatic segmentation of tissues on WSIs  

Pathologists commonly work with Whole-Slide Images (WSIs) digitized from histopathological sections cut from tumour biopsies, stained for H&E and different immunohistochemistry (IHC) biomarkers. The gigapixels images are stored as pyramid-structured objects and offer pathological insights at various magnification levels, from the molecular to the macroscopic scales. 
Throughout their analysis, experts perform segmentation tasks, grade assessment or subtyping at different scales, zooming in and out to grasp information and context from tissues. Despite these operations are instrumental in tumour diagnosis and prognosis, they are time-consuming and might variate among observers. For these reasons, Deep Learning-based frameworks emerged as potential tools to alleviate pathologists workload and cope with tissue classification and segmentation on high-resolution histopatholgical images.

🔗 https://www.sciencedirect.com/science/article/pii/S2001037024004057#fig0005 \
🔗 https://www.sciencedirect.com/science/article/abs/pii/S0010482521005242

<img src='Figures/Automatic segmentation illu.jpg' width='100%'> 

# ⛽ Sparse manual annotations scenario

The Deep Learning-based frameworks provide automatic segmentation tools, relying on the training of neural networks still based on manual annotations of WSI tissues. Learning transfer strategies appear as a good compromise for training model to classify pixels on WSI-inferred images, in sparse manual annotations scenario. \
From a few regions with segmented compartments, distinguising tissues with different properties, a pre-trained neural network can be optimized for a specific task. Indeed, extensively used Deep Learning architectures like DenseNet or SegNet have been used for tumour segmentation tasks. The general outline is the following: 

1. Manual annotations - Outline pre-determined distinct compartments, resulting in semantic masks.  
2. Model fine-tuning - Train a neural network with a dataset for the specific task. 

<img src='Figures/Sparse annotations scenario illu.jpg' width='100%'> 

# 🧩 Approach on H&E tiles from ccRCC 

📡 We propose a case study on H&E tiles derived from resected ccRCC tumours WSIs. The pre-trained network DenseNet169 is employed. \
  🔗 https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet169.html

🔬 The segmentation task here consists of identifying the neoplastic cells (blue) from the tumour micro-environment (red) and the background (black).  

<img src='Figures/Approach on ccRCC tiles illu.jpg' width='100%'> 

DenseNet model fine-tuning for ccRCC tumour semantic segmentation 
