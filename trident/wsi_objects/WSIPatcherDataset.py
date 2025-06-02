from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
class WSIPatcherDataset(Dataset):
    """ Dataset from a WSI patcher to directly read tiles on a slide  """
    
    def __init__(self, patcher, transform):
        self.patcher = patcher
        self.transform = transform
                              
    def __len__(self):
        return len(self.patcher)
    
    def __getitem__(self, index):
        tile, x, y = self.patcher[index]

        if self.transform:
            tile = self.transform(tile)
        else:
            tile = torch.from_numpy(np.array(tile))

        return tile, (x, y)
    
class WSIPatchesDataset(Dataset):
    """ Dataset from a WSI patcher to directly read tiles on a slide  """
    
    def __init__(self, patches, transform):
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        patch = self.patches[index]
        if isinstance(patch, np.ndarray):
            if patch.dtype != np.uint8:
                patch = (patch * 255).astype(np.uint8) if patch.max() <= 1.0 else patch.astype(np.uint8)
            patch = Image.fromarray(patch)
        patch = self.transform(patch)
        return patch
