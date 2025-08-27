import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, Union
import os 
from shapely import Polygon, MultiPolygon


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create the heatmap overlay based on scores and coordinates.
    
    Args:
        scores (np.ndarray): Normalized scores.
        coords (np.ndarray): Coordinates of patches.
        patch_size_level0 (int): Patch size at level 0.
        scale (np.ndarray): Scaling factors.
        region_size (Tuple[int, int]): Dimensions of the region.
    
    Returns:
        np.ndarray: Heatmap overlay.
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords = np.ceil(coords * scale).astype(int)
    
    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)
    
    for idx, coord in enumerate(coords):
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += scores[idx]
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1
    
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # Set areas with no data to NaN
    
    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply a colormap to the heatmap overlay.
    
    Args:
        overlay (np.ndarray): Heatmap overlay.
        cmap_name (str): Colormap name.

    Returns:
        np.ndarray: Colored overlay image.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid
    return overlay_colored


def visualize_heatmap(
    wsi,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    vis_level: Optional[int] = 2,
    cmap: str = 'coolwarm',
    normalize: bool = True,
    num_top_patches_to_save: int = -1,
    output_dir: Optional[str] = "output",
    vis_mag: Optional[int] = None,
    overlay_only = False,
    filename = 'heatmap.png'
) -> str:
    """
    Generate a heatmap visualization overlayed on a whole slide image (WSI).
    
    Args:
        wsi: Whole slide image object.
        scores (np.ndarray): Scores associated with each coordinate.
        coords (np.ndarray): Coordinates of patches at level 0.
        patch_size_level0 (int): Patch size at level 0.
        vis_level (Optional[int]): Visualization level.
        cmap (str): Colormap to use for the heatmap.
        normalize (bool): Whether to normalize the scores.
        num_top_patches_to_save (int): Number of high-score patches to save. If set to -1, do not save any. Defaults to -1.
        output_dir (Optional[str]): Directory to save heatmap and top-k patches.
        vis_mag (Optional[int]): Visualization Magnification. This will overwrite vis_level
        overlay_only bool: Whenever to save the overlay only. If set to True, save the overlay on top of downscaled version of the WSI. Defaults to False.
        filename (str): file will be saved in `output_dir`/`filename`

    Returns:
        str: Path to the saved heatmap image.
    """

    if normalize:
        from scipy.stats import rankdata
        scores = rankdata(scores, 'average') / len(scores) * 100 / 100
    
    if vis_mag is None:
        downsample = wsi.level_downsamples[vis_level]
    else:
        src_mag = wsi.mag
        downsample = src_mag / vis_mag
        if not overlay_only:
            vis_level, _ = wsi.get_best_level_and_custom_downsample(downsample)
    
    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    overlay = create_overlay(scores, coords, patch_size_level0, scale, region_size)

    overlay_colored = apply_colormap(overlay, cmap)
    
    if overlay_only:
        blended_img = overlay_colored
    else:
        img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
        img = np.array(img)
        
        blended_img = cv2.addWeighted(img, 0.6, overlay_colored, 0.4, 0)
    
    blended_img = Image.fromarray(blended_img)

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, filename)
    blended_img.save(heatmap_path)

    if num_top_patches_to_save > 0:
        topk_dir = os.path.join(output_dir, "topk_patches")
        os.makedirs(topk_dir, exist_ok=True)
        topk_indices = np.argsort(scores)[-num_top_patches_to_save:]
        for idx, i in enumerate(topk_indices):
            x, y = coords[i]
            patch = wsi.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
            patch.save(os.path.join(topk_dir, f"top_{idx}_score_{scores[i]:.4f}.png"))

    return heatmap_path



def _visualize_coords(wsi, width, height, patch_size_src, xy_iterator, overlay_only, rgba):
    max_dimension = 1000
    if width > height:
        thumbnail_width = max_dimension
        thumbnail_height = int(thumbnail_width * height / width)
    else:
        thumbnail_height = max_dimension
        thumbnail_width = int(thumbnail_height * width / height)

    downsample_factor = width / thumbnail_width

    thumbnail_patch_size = max(1, int(patch_size_src / downsample_factor))

    # Get thumbnail in right format
    if overlay_only:
        if rgba:
            canvas = np.zeros((thumbnail_height, thumbnail_width, 4)).astype(np.uint8)
        else:
            canvas = np.zeros((thumbnail_height, thumbnail_width, 3)).astype(np.uint8)
    else:
        canvas = np.array(wsi.get_thumbnail((thumbnail_width, thumbnail_height))).astype(np.uint8)

    color = (255, 0, 0, 255) if rgba else (255, 0, 0)

    # Draw rectangles for patches
    for (x, y) in xy_iterator:
        x, y = int(x/downsample_factor), int(y/downsample_factor)
        thickness = max(1, thumbnail_patch_size // 10)
        canvas = cv2.rectangle(
            canvas, 
            (x, y), 
            (x + thumbnail_patch_size, y + thumbnail_patch_size), 
            color, 
            thickness
        )

    return canvas


def visualize_coords_overlay(width, height, patch_size_src, xy_iterator, rgba):
    canvas = _visualize_coords(None, width, height, patch_size_src, xy_iterator,
                               overlay_only=True, rgba=rgba)
    
    return Image.fromarray(canvas)

def visualize_coords_with_thumbnail(wsi, patch_size_src, xy_iterator, 
    dst_mag, dst_pixel_size, patch_size_target, overlap):

    canvas = _visualize_coords(wsi, wsi.width, wsi.height, patch_size_src, xy_iterator,
                               overlay_only=False, rgba=False)
    # Add annotations
    text_area_height = 130
    text_x_offset = int(canvas.shape[1] * 0.03)  # Offset as 3% of thumbnail width
    text_y_spacing = 25  # Vertical spacing between lines of text

    canvas[:text_area_height, :300] = (
        canvas[:text_area_height, :300] * 0.5
    ).astype(np.uint8)

    patch_mpp_mag = f"{dst_mag}x" if dst_mag is not None else f"{dst_pixel_size}um/px"

    cv2.putText(canvas, f'{len(len(xy_iterator))} patches', (text_x_offset, text_y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(canvas, f'width={wsi.width}, height={wsi.height}', (text_x_offset, text_y_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(canvas, f'mpp={wsi.mpp}, mag={wsi.mag}', (text_x_offset, text_y_spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f'patch={patch_size_target} w. overlap={overlap} @ {patch_mpp_mag}', (text_x_offset, text_y_spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return Image.fromarray(canvas)