# 🎯 Automatic segmentation of tissues on WSIs  

Pathologists commonly work with Whole-Slide Images (WSIs) digitized from histopathological sections cut from tumour biopsies, stained for H&E and different immunohistochemistry (IHC) biomarkers. The gigapixels images are stored as pyramid-structured objects and offer pathological insights at various magnification levels, from the molecular to the macroscopic scales. 
Throughout their analysis, experts perform segmentation tasks, grade assessment or subtyping at different scales, zooming in and out to grasp information and context from tissues. Despite these operations are instrumental in tumour diagnosis and prognosis, they are time-consuming and might variate among observers. For these reasons, Deep Learning-based frameworks emerged as potential tools to alleviate pathologists workload and cope with tissue classification and segmentation on high-resolution histopatholgical images.\

🔗 https://www.sciencedirect.com/science/article/pii/S2001037024004057#fig0005 \
🔗 https://www.sciencedirect.com/science/article/abs/pii/S0010482521005242

<img src='Figures/Source image.jpg' width='100%'> 

The initial output of the deep-learning pipeline involves classifying tissue tiles as either tumor or non-tumor based on their morphological features. A: annotations made by the pathologist on the H&E slide indicate the tumor area manually identified across the image. B: the algorithm's tumor classification is shown, with red highlighting the regions where the model has detected tumor-containing tiles, visually marking the zones identified by the deep-learning process.

# ⛽ Sparse manual annotations scenario


# 🧩 Approach on H&E tiles from ccRCC 


DenseNet model fine-tuning for ccRCC tumour semantic segmentation 
