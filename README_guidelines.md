# 🎨 Manual tiles annotation

Our approach was conducted on ccRCC Whole-Slide Images from the TCGA database, a repository and computational platform for cancer researchers with thousands of kidney cancer histopathological images. \
🔗 https://portal.gdc.cancer.gov/analysis_page?app= \
A collection of 2048x2048 tiles was manually annotated and reviewed by an expert pathologist, distinguishing pixels into three categories: neoplastic (blue), non neoplastic (red) or background (black). This operation was performed using the application Sketchbook. 

<img src='Figures/Pipeline 1.jpg' width='100%'> 

# 🧹 Data refinement 

The semantic masks delineating compartments cannot be used as direct inputs as they comprise 3 channels and may represent colors with close but distinct RGB tuple (R, G, B) in [0, 255]. The colour assigned to each compartment should be converted into semantic labels (0: background; 1: non neoplastic; 2: neoplastic). To that end, pixels channels are summed and associated with a label, while pixels presenting an unreferenced sum are assigned with the closest color for consistency. 

# 🗃️ PyTorch dataset integration

```python

```

# 🚀 Data loader creation

```python

```
 
# 🏗️ Neural architecture design

```python

```

<img src='Figures/Pipeline 2.jpg' width='100%'> 
 
# ⛳ Model fine-tuning

```python

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
 
