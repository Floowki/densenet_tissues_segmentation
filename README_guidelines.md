# 🎨 Manual tiles annotation

Our approach was conducted on ccRCC Whole-Slide Images from the TCGA database, a repository and computational platform for cancer researchers with thousands of kidney cancer histopathological images. 
🔗 https://portal.gdc.cancer.gov/analysis_page?app= \
A collection of 512x512 tiles was manually annotated and reviewed by an expert pathologist, distinguishing pixels into three categories: neoplastic (blue), non neoplastic (red) or background (black). This operation was performed using the application Sketchbook. 

<img src='Figures/Pipeline 1.jpg' width='100%'> 

# 🧹 Data refinement 

```python

```

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
 
