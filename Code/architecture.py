import torch.nn.functional as F 
from torchvision import models  
import torch.nn as nn 


#### DENSENET 169 ####
class CustomDenseNet(nn.Module):
    def __init__(self, output_size=512):
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
        self.output_size = output_size

    def forward(self, x):
        
        # Extract features from the backbone
        features = self.densenet.features(x)

        # Apply the segmentation head
        output = self.upsample(features)

        # Ensure output has the correct spatial dimensions
        if output.shape[2] != self.output_size or output.shape[3] != self.output_size:
            output = F.interpolate(output, size=(self.output_size, self.output_size), mode='bilinear', align_corners=True)

        return output