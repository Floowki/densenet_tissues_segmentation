import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def visualize_seg(source_path, semantic_path, correct_colors) : 
    #| Args :  
    #|   # source_path : path to a source image                   
    #|   # semantic_path : path to a semantic mask (algo result)       

    #| Outputs : 
    #|   # display the source / segmented images 
     
    # Load the source and semantic images
    img_source = cv2.imread(source_path) 
    img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)

    predicted = cv2.imread(semantic_path) 
    predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)
    
    rgb_semantic = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype="uint8")

    rgb_semantic[predicted == 0] = correct_colors[0]    # background
    rgb_semantic[predicted == 1] = correct_colors[1]    # stroma
    rgb_semantic[predicted == 2] = correct_colors[2]    # cellular

    # Plot the source image and the resulting semantic segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Source image
    axes[0].imshow(img_source)
    axes[0].set_title('Source Image')
    axes[0].axis('off')

    # Resulting semantic segmentation
    axes[1].imshow(rgb_semantic)
    axes[1].set_title('Semantic Segmentation')
    axes[1].axis('off')

    plt.show()
    
    return 