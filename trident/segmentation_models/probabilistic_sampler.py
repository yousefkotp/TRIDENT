import random
import numpy as np
import os
from typing import Dict, Any, List

class ProbabilisticSampler:
    """Probabilistic sampler for generating random subpatches from patches."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_subpatches = config.get("min_subpatches")
        self.max_subpatches = config.get("max_subpatches")
        self.subpatch_size_min = config.get("subpatch_size_min")
        self.subpatch_size_max = config.get("subpatch_size_max")
        self.sampling_distribution = config.get("sampling_distribution")
        self.poisson_lambda = config.get("poisson_lambda", 3.0)
        self.geometric_p = config.get("geometric_p", 0.3)
        self.debug = config.get("debug", None) is not None
        self.debug_dir = config.get("debug", None)
        
    def sample_num_subpatches(self) -> int:
        """Sample the number of subpatches to extract."""
        if self.sampling_distribution == "uniform":
            return random.randint(self.min_subpatches, self.max_subpatches)
        elif self.sampling_distribution == "poisson":
            num = np.random.poisson(self.poisson_lambda)
            return np.clip(num, self.min_subpatches, self.max_subpatches)
        elif self.sampling_distribution == "geometric":
            num = np.random.geometric(self.geometric_p)
            return np.clip(num, self.min_subpatches, self.max_subpatches)
        else:
            raise ValueError(f"Unknown sampling distribution: {self.sampling_distribution}")
    
    def sample_subpatches(self, img: np.ndarray) -> List[np.ndarray]:
        """Sample random subpatches from the input image."""
        if len(img.shape) != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {img.shape}")
            
        img_h, img_w = img.shape[:2]
        num_subpatches = self.sample_num_subpatches()
        subpatches = []
        
        for _ in range(num_subpatches):
            # Random subpatch size
            subpatch_size = random.randint(self.subpatch_size_min, self.subpatch_size_max)
            
            # Ensure subpatch fits within image
            max_x = max(0, img_w - subpatch_size)
            max_y = max(0, img_h - subpatch_size)
            
            if max_x <= 0 or max_y <= 0:
                continue  # Skip if subpatch is larger than image
                
            # Random position
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Extract subpatch
            subpatch = img[y:y+subpatch_size, x:x+subpatch_size]
            
            # Apply intensity filtering if enabled
            mean_intensity = np.mean(subpatch)
            if mean_intensity <= 5 or mean_intensity >= 210:
                continue
            
            subpatches.append(subpatch)
        
        # Debug visualization if enabled
        if self.debug and subpatches:
            self._save_debug_visualization(img, subpatches)
        
        return subpatches
    
    def _save_debug_visualization(self, original_img: np.ndarray, subpatches: List[np.ndarray]):
        """Save debug visualization showing original image and sampled subpatches."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, min(len(subpatches) + 1, 6), figsize=(15, 3))
        if len(subpatches) == 0:
            return
            
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Show original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Patch')
        axes[0].axis('off')
        
        # Show subpatches
        for i, subpatch in enumerate(subpatches[:5]):  # Show max 5 subpatches
            if i + 1 < len(axes):
                axes[i + 1].imshow(subpatch)
                axes[i + 1].set_title(f'Subpatch {i+1}')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        os.makedirs(self.debug_dir, exist_ok=True)
        image_name = f'prob_sampling_{random.randint(0, 1000000)}.png'
        plt.savefig(f'{self.debug_dir}/{image_name}')
        plt.close()