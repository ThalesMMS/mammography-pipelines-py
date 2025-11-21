import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # Load ResNet50 with default weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        base = models.resnet50(weights=weights)
        
        # Remove FC
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.features.eval()
        self.features.to(device)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x

    def extract(self, loader: DataLoader) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        embeddings = []
        metadata = []
        
        for batch in tqdm(loader, desc="Extracting features"):
            if len(batch) >= 3:
                imgs, _, metas = batch[0], batch[1], batch[2]
            else:
                imgs = batch[0] # If simplistic loader
                metas = []
                
            imgs = imgs.to(self.device)
            
            feats = self.forward(imgs)
            embeddings.append(feats.cpu().numpy())
            
            if isinstance(metas, list):
                metadata.extend(metas)
                
        if not embeddings:
            return np.array([]), []
            
        return np.concatenate(embeddings, axis=0), metadata
