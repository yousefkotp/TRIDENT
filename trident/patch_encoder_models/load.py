import traceback
from abc import abstractmethod
import torch
import os 

from trident.patch_encoder_models.utils.constants import get_constants
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms
from trident.IO import get_weights_path, has_internet_connection

"""
This file contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""

def encoder_factory(model_name, **kwargs):
    """
    Build a patch encoder model.

    Args:   
        model_name (str): The name of the model to build.
        **kwargs: Additional arguments to pass to the encoder constructor.

    Returns:
        torch.nn.Module: The patch encoder model.
    """

    if model_name == 'conch_v1':
        enc = Conchv1InferenceEncoder
    elif model_name == 'conch_v15':
        enc = Conchv15InferenceEncoder
    elif model_name == 'uni_v1':
        enc = UNIInferenceEncoder
    elif model_name == 'uni_v2':
        enc = UNIv2InferenceEncoder
    elif model_name == 'ctranspath':
        enc = CTransPathInferenceEncoder
    elif model_name == 'phikon':
        enc = PhikonInferenceEncoder
    elif model_name == 'resnet50':
        enc = ResNet50InferenceEncoder
    elif model_name == 'gigapath':
        enc = GigaPathInferenceEncoder
    elif model_name == 'virchow':
        enc = VirchowInferenceEncoder
    elif model_name == 'virchow2':
        enc = Virchow2InferenceEncoder
    elif model_name == 'hoptimus0':
        enc = HOptimus0InferenceEncoder
    elif model_name == 'hoptimus1':
        enc = HOptimus1InferenceEncoder
    elif model_name == 'phikon_v2':
        enc = Phikonv2InferenceEncoder
    elif model_name == 'musk':
        enc = MuskInferenceEncoder
    elif model_name == 'hibou_l':
        enc = HibouLInferenceEncoder
    elif model_name == 'kaiko-vitb8':
        enc = KaikoB8InferenceEncoder
    elif model_name == 'kaiko-vitb16':
        enc = KaikoB16InferenceEncoder
    elif model_name == 'kaiko-vits8':
        enc = KaikoS8InferenceEncoder
    elif model_name == 'kaiko-vits16':
        enc = KaikoS16InferenceEncoder
    elif model_name == 'kaiko-vitl14':
        enc = KaikoL14InferenceEncoder
    elif model_name == 'lunit-vits8':
        enc = LunitS8InferenceEncoder
    else:
        raise ValueError(f"Unknown encoder name {model_name}")

    return enc(**kwargs)

####################################################################################################

class BasePatchEncoder(torch.nn.Module):

    _has_internet = has_internet_connection()
    
    def __init__(self, **build_kwargs):
        super().__init__()
        self.enc_name = None
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)
    
    def ensure_valid_weights_path(self, weights_path):
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")
    
    def ensure_has_internet(self, enc_name):
        if not BasePatchEncoder._has_internet:
            raise FileNotFoundError(
                f"Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                f"and place it in the model registry in:\n`trident/patch_encoder_models/local_ckpts.json`"
            )

    def forward(self, x):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class CustomInferenceEncoder(BasePatchEncoder):
    def __init__(self, enc_name, model, transforms, precision):
        super().__init__()
        self.enc_name = enc_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        
    def _build(self):
        return None, None, None


class MuskInferenceEncoder(BasePatchEncoder):
    
    def _build(self, inference_aug=False, with_proj=False, out_norm=False, return_global=True):
        '''
        Args:
            inference_aug (bool): Whether to use test-time multiscale augmentation. Default is False to allow for fair comparison with other models.
        '''
        import timm
        
        self.enc_name = 'musk'
        self.inference_aug = inference_aug
        self.with_proj = with_proj
        self.out_norm = out_norm
        self.return_global = return_global
    
        try:
            from musk import utils, modeling
        except:
            traceback.print_exc()
            raise Exception("Please install MUSK `pip install fairscale git+https://github.com/lilab-stanford/MUSK`")

        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            raise NotImplementedError("MUSK doesn't support local model loading. PR welcome!")
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("musk_large_patch16_384")
                utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            except:
                traceback.print_exc()
                raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")
        
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import InterpolationMode
        eval_transform = get_eval_transforms(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, target_img_size = 384, center_crop = True, interpolation=InterpolationMode.BICUBIC, antialias=True)
        precision = torch.float16
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model(
                image=x,
                with_head=self.with_proj,
                out_norm=self.out_norm,
                ms_aug=self.inference_aug,
                return_global=self.return_global  
                )[0]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.


class Conchv1InferenceEncoder(BasePatchEncoder):

    def _build(self, with_proj=False, normalize=False):
        self.enc_name = 'conch_v1'
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="hf_hub:MahmoodLab/conch")
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1 model, make sure that you were granted access and that you correctly registered your token")
    
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=self.with_proj, normalize=self.normalize)
    

class CTransPathInferenceEncoder(BasePatchEncoder):

    def _build(self):
        from torchvision.transforms import InterpolationMode
        from torch import nn

        try:
            from .model_zoo.ctranspath.ctran import ctranspath
        except:
            traceback.print_exc()
            raise Exception("Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`")
        
        self.enc_name = 'ctranspath'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                from huggingface_hub import hf_hub_download   
                weights_path = hf_hub_download(
                    repo_id="MahmoodLab/hest-bench",
                    repo_type="dataset",
                    filename="CHIEF_CTransPath.pth",
                    subfolder="fm_v1/ctranspath",
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download CTransPath model, make sure that you were granted access and that you correctly registered your token")

        state_dict = torch.load(weights_path, weights_only=True)['model']
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys found in state dict: {unexpected}"
        assert missing == ['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask'], f"Unexpected missing keys: {missing}"

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        
        return model, eval_transform, precision


class PhikonInferenceEncoder(BasePatchEncoder):

    def _build(self):
        from transformers import ViTModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'phikon'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            model_dir = os.path.dirname(weights_path)
            model = ViTModel.from_pretrained(model_dir, add_pooling_layer=False, local_files_only=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class HibouLInferenceEncoder(BasePatchEncoder):

    def _build(self):
        from transformers import AutoModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'hibou_l'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model_dir = os.path.dirname(weights_path)
            required_files = ["model.safetensors", "config.json"]
            missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required file(s): {', '.join(missing_files)} in {model_dir}. "
                    f"These can be downloaded from https://huggingface.co/histai/hibou-L"
                )
            model = AutoModel.from_pretrained(model_dir)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Hibou-L model, make sure that you were granted access and that you correctly registered your token")
        
        mean, std = get_constants('hibou')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.pooler_output
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out


class KaikoInferenceEncoder(BasePatchEncoder):
    MODEL_NAME = None  # set in subclasses
    HF_HUB_ID = None # set in subclasses
    IMG_SIZE = None

    def _build(self):
        import timm
        from torchvision.transforms import InterpolationMode
        self.enc_name = f"kaiko-{self.MODEL_NAME}"
        weights_path = get_weights_path("patch", self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model = timm.create_model(
                f"{self.HF_HUB_ID}",
                num_classes=0,
                checkpoint_path=weights_path,
                img_size=self.IMG_SIZE,
                dynamic_img_size=True
            )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model(
                    model_name=f"hf-hub:1aurent/{self.HF_HUB_ID}.kaiko_ai_towards_large_pathology_fms",
                    dynamic_img_size=True,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.IMG_SIZE,
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download Kaiko model.")

        mean, std = get_constants("kaiko")
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class KaikoS16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits16"
    HF_HUB_ID = "vit_small_patch16_224"
    IMG_SIZE = 224


class KaikoS8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits8"
    HF_HUB_ID = "vit_small_patch8_224"
    IMG_SIZE = 224


class KaikoB16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb16"
    HF_HUB_ID = "vit_base_patch16_224"
    IMG_SIZE = 224


class KaikoB8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb8"
    HF_HUB_ID = "vit_base_patch8_224"
    IMG_SIZE = 224


class KaikoL14InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitl14"
    HF_HUB_ID = "vit_large_patch14_reg4_dinov2"
    IMG_SIZE = 518


class ResNet50InferenceEncoder(BasePatchEncoder):
    def _build(
        self, 
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        img_size=224,
        pool=True
    ):
        import timm
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'resnet50'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model = timm.create_model("resnet50", pretrained=False, pretrained_cfg_overlay=dict(file=weights_path), **timm_kwargs)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download ResNet50 model.")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class LunitS8InferenceEncoder(BasePatchEncoder):

    def _build(self):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'lunit-vits8'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {"img_size": 224}
            model = timm.create_model("vit_small_patch8_224", checkpoint_path=weights_path, **timm_kwargs)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Lunit S8 model, make sure that you were granted access and that you correctly registered your token.")

        data_config = resolve_model_data_config(model)
        eval_transform = create_transform(**data_config, is_training=False)
        precision = torch.float32

        return model, eval_transform, precision
    

class UNIInferenceEncoder(BasePatchEncoder):
    def _build(
        self, 
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'uni_v1'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model = timm.create_model("vit_large_patch16_224", **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision
    

class UNIv2InferenceEncoder(BasePatchEncoder):

    def _build(self):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        self.enc_name = 'uni_v2'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }

        if weights_path:
            model = timm.create_model(model_name='vit_giant_patch14_224', pretrained=False, **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.bfloat16
        return model, eval_transform, precision
    

class GigaPathInferenceEncoder(BasePatchEncoder):

    def _build(
        self, 
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'gigapath'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {
                "img_size": 224,
                "in_chans": 3,
                "patch_size": 16,
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "mlp_ratio": 5.33334,
                "num_classes": 0
            }
            model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision

    
class VirchowInferenceEncoder(BasePatchEncoder):
    import timm
    
    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'virchow'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {
                "img_size": 224,
                "init_values": 1e-5,
                "num_classes": 0,
                "mlp_ratio": 5.3375,
                "global_pool": "",
                "dynamic_img_size": True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
            }
            model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings


class Virchow2InferenceEncoder(BasePatchEncoder):
    import timm
    
    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'virchow2'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {
                "img_size": 224,
                "init_values": 1e-5,
                "num_classes": 0,
                "reg_tokens": 4,
                "mlp_ratio": 5.3375,
                "global_pool": "",
                "dynamic_img_size": True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
            }
            model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow-2 model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding


class HOptimus0InferenceEncoder(BasePatchEncoder):

    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus0'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {
                "num_classes": 0,
                "img_size": 224,
                "global_pool": "token",
                'init_values': 1e-5,
                'dynamic_img_size': False
            }
            model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-0 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class HOptimus1InferenceEncoder(BasePatchEncoder):
    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
        **kwargs
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            timm_kwargs = {
                "num_classes": 0,
                "img_size": 224,
                "global_pool": "token",
                'init_values': 1e-5,
                'dynamic_img_size': False
            }
            model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-1 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class Phikonv2InferenceEncoder(BasePatchEncoder):

    def _build(self):
        from transformers import AutoModel
        import torchvision.transforms as T
        from .utils.constants import IMAGENET_MEAN, IMAGENET_STD

        self.enc_name = 'phikon_v2'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        
        if weights_path:
            model_dir = os.path.dirname(weights_path)
            required_files = ["model.safetensors", "config.json"]
            missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required file(s): {', '.join(missing_files)} in {model_dir}. "
                    f"These can be downloaded from https://huggingface.co/owkin/phikon-v2"
                )
            model = AutoModel.from_pretrained(model_dir)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("owkin/phikon-v2")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = T.Compose([
            T.Resize(224),  
            T.CenterCrop(224),  
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalize with specified mean and std
        ])

        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.model(x)
        out = out.last_hidden_state[:, 0, :]
        return out


class Conchv15InferenceEncoder(BasePatchEncoder):

    def _build(self, img_size=448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

        self.enc_name = 'conch_v15'
        weights_path = get_weights_path('patch', self.enc_name)
        self.ensure_valid_weights_path(weights_path)

        if weights_path:
            model, eval_transform = create_model_from_pretrained(checkpoint_path=weights_path, img_size=img_size)
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")

        precision = torch.float16
        return model, eval_transform, precision

