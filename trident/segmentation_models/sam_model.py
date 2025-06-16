from typing import List, Tuple, Optional, Dict, Any, Literal, Union
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import os
import random


class SamModelLoader:
    """
    Handles loading and initialization of different SAM model versions.
    Supports both original SAM and SAM 2.0 models.
    """
    
    def __init__(
        self, 
        model_type: str = "vit_h", 
        checkpoint_path: Optional[str] = None,
        sam_version: Literal["sam", "sam2"] = "sam",
        device: str = "cuda",
        pred_iou_thresh: float = 0.3,
        stability_score_thresh: float = 0.6,
        model_cfg: Optional[str] = None,  # (only SAM-v2)
    ):
        """
        Initialize a SAM model loader.
        
        Args:
            model_type: Model architecture type (e.g., "vit_h", "vit_b")
            checkpoint_path: Path to the model checkpoint file
            sam_version: SAM version to use, either "sam" for original or "sam2" for SAM 2.0
            device: Device to load the model on
            pred_iou_thresh: Prediction IoU threshold for mask filtering
            stability_score_thresh: Stability score threshold for mask filtering
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_version = sam_version
        self.device = device
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.model = None
        self.mask_generator = None
        self.model_cfg = model_cfg
        
    def load_model(self) -> None:
        """
        Load the specified SAM model version.
        """
        if self.model is not None:
            return
            
        if self.sam_version == "sam":
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                
                if self.checkpoint_path is None:
                    raise ValueError("Checkpoint path must be provided for SAM model")
                    
                self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.model.eval()
                self.model.to(self.device)
                self.model = torch.compile(self.model, mode="reduce-overhead")
                torch.backends.cuda.matmul.allow_tf32 = True
                
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.model,
                    pred_iou_thresh=self.pred_iou_thresh,
                    stability_score_thresh=self.stability_score_thresh
                )
                print("SAM model loaded")
            except ImportError:
                raise ImportError("segment_anything package not found. Install it with: pip install segment_anything")
                
        elif self.sam_version == "sam2":

            if self.checkpoint_path is None or self.model_cfg is None:
                raise ValueError(
                    "`checkpoint_path` **and** `model_cfg` must be provided for SAM 2"
                )

            self.model = build_sam2(self.model_cfg, self.checkpoint_path)
            self.model.eval()
            self.model.to(self.device)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            torch.backends.cuda.matmul.allow_tf32 = True

            self.mask_generator = SAM2AutomaticMaskGenerator(
                self.model,
                points_per_side=32,
                points_per_batch=256,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.95,
                stability_score_offset=1.0,
                mask_threshold=0.0,
                box_nms_thresh=0.7,
                crop_n_layers=0,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=512 / 1500,
                crop_n_points_downscale_factor= 1,
                min_mask_region_area=0,
                use_m2m=False,
                multimask_output=True,
            )
            print("SAM 2.0 model loaded successfully")
        else:
            raise ValueError(f"Unsupported SAM version: {self.sam_version}. Use 'sam' or 'sam2'.")

    def generate_masks(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate masks for the provided image.
        
        Args:
            image: Input image (numpy array)
            **kwargs: Additional arguments to pass to the mask generator
            
        Returns:
            List of dictionaries containing mask data
        """

        """Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

            segmentation : the mask
            area : the area of the mask in pixels
            bbox : the boundary box of the mask in XYWH format
            predicted_iou : the model's own prediction for the quality of the mask
            point_coords : the sampled input point that generated this mask
            stability_score : an additional measure of mask quality
            crop_box : the crop of the image used to generate this mask in XYWH format
        """
        if self.model is None:
            self.load_model()
        with torch.inference_mode():
            if self.sam_version == "sam":
                return self.mask_generator.generate(image, **kwargs)
            elif self.sam_version == "sam2":
                return self.mask_generator.generate(image, **kwargs)
            else:
                raise ValueError(f"Unsupported SAM version: {self.sam_version}. Use 'sam' or 'sam2'.")
    
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
