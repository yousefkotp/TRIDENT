from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_eval_transforms(mean, std, target_img_size = -1, center_crop = False, **resize_kwargs):
    trsforms = []
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size, **resize_kwargs))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


def get_vit_val_transforms(normalize: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Validation-time transforms for ViT-style encoders.
    """
    transforms_list = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]

    if normalize:
        transforms_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(transforms_list)
