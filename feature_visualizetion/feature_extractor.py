import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List, Optional, Union, Dict

class FeatureExtractor:
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        
        # Load model
        if hasattr(models, model_name):
            # Handle weights parameter for newer torchvision versions
            try:
                weights = 'DEFAULT' if pretrained else None
                self.model = getattr(models, model_name)(weights=weights)
            except TypeError:
                # Fallback for older torchvision
                self.model = getattr(models, model_name)(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model.to(device)
        self.model.eval()
        
        # Define return nodes for supported architectures
        self.node_mapping = self._get_node_mapping(model_name)
        
        # Create feature extractor if mapping exists
        if self.node_mapping:
            self.extractor = create_feature_extractor(self.model, return_nodes=self.node_mapping)
        else:
            self.extractor = self.model

    def _get_node_mapping(self, model_name: str) -> Dict[str, str]:
        """
        Get mapping from model layer names to output names (layer0, layer1, etc.)
        """
        if 'resnet' in model_name:
            return {
                'layer1': '0',
                'layer2': '1',
                'layer3': '2',
                'layer4': '3',
            }
        elif 'mobilenet_v2' in model_name:
            # MobileNetV2 inverted residual blocks
            return {
                'features.3': '0',
                'features.6': '1',
                'features.13': '2',
                'features.18': '3',
            }
        elif 'vgg' in model_name:
            # VGG features (after max pooling layers usually)
            # VGG16: 4, 9, 16, 23, 30 (maxpools)
            return {
                'features.9': '0',
                'features.16': '1',
                'features.23': '2',
                'features.30': '3',
            }
        # Add more models as needed
        return {}

    def extract_features(
        self, 
        image: torch.Tensor, 
        layer_indices: Optional[List[int]] = None,
        normalize: bool = True,
        reshape: bool = True
    ) -> List[torch.Tensor]:
        """
        Extract features from the model.
        
        Args:
            image (torch.Tensor): Input image tensor (B, C, H, W) or (C, H, W)
            layer_indices (List[int]): Indices of layers to return (0-3 for ResNet)
            normalize (bool): Whether to normalize features (not implemented here, kept for API compatibility)
            reshape (bool): Whether to reshape features (not implemented here, kept for API compatibility)
            
        Returns:
            List[torch.Tensor]: List of feature tensors
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        image = image.to(self.device)
        
        if self.node_mapping:
            outputs = self.extractor(image)
            # Convert dictionary values to list, sorted by key (which are '0', '1', etc.)
            features = [outputs[str(i)] for i in range(len(self.node_mapping)) if str(i) in outputs]
        else:
            # Fallback for unsupported models - just return output
            features = [self.model(image)]
            
        # Filter by layer_indices if provided
        if layer_indices is not None:
            selected_features = []
            for idx in layer_indices:
                if 0 <= idx < len(features):
                    selected_features.append(features[idx])
            return selected_features
        
        return features
