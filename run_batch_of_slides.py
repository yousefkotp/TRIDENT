"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""

import argparse
import torch
from trident import Processor


def build_parser():
    """
    Parse command-line arguments for the Trident processing script.
    """
    parser = argparse.ArgumentParser(description='Run Trident')
    # Generic arguments 
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for processing tasks')
    parser.add_argument('--task', type=str, default='seg', 
                        choices=['cache', 'seg', 'coords', 'feat', 'all'], 
                        help='Task to run: cache, seg (segmentation), coords (save tissue coordinates), img (save tissue images), feat (extract features)')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory to store outputs')
    parser.add_argument('--wsi_cache', type=str, default=None, 
                        help='Directory to copy slides to for local processing')
    parser.add_argument('--clear_cache', action='store_true', default=False, 
                        help='Delete slides from cache after processing')
    parser.add_argument('--skip_errors', action='store_true', default=False, 
                        help='Skip errored slides and continue processing')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers. Set to 0 to use main process.')

    # Slide-related arguments
    parser.add_argument('--wsi_dir', type=str, required=True, 
                        help='Directory containing WSI files (no nesting allowed)')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--custom_list_of_wsis', type=str, default=None,
                    help='Custom list of WSIs specified in a csv file.')
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim'], default=None,
                    help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim"]. Defaults to None (auto-determine which reader to use).')
    
    # Segmentation arguments 
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc'], 
                        help='Type of tissue vs background segmenter. Options are HEST or GrandQC.')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                    help='Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='Do you want to remove holes?')
    parser.add_argument('--remove_artifacts', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove artifacts (including penmarks, blurs, stains, etc.)?')
    parser.add_argument('--remove_penmarks', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove penmarks?')
    # Patching arguments
    parser.add_argument('--mag', type=int, choices=[5, 10, 20, 40, 80], default=20, 
                        help='Magnification for coords/features extraction')
    parser.add_argument('--patch_size', type=int, default=512, 
                        help='Patch size for coords/image extraction')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0. ')
    parser.add_argument('--min_tissue_proportion', type=float, default=0., 
                        help='Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0. ')
    parser.add_argument('--coords_dir', type=str, default=None, 
                        help='Directory to save/restore tissue coordinates')
    # Feature extraction arguments 
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=['conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon', 
                                 'resnet50', 'gigapath', 'virchow', 'virchow2', 
                                 'hoptimus0', 'hoptimus1', 'phikon_v2', 'conch_v15', 'musk', 'hibou_l',
                                 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitb8', 'kaiko-vitb16',
                                 'kaiko-vitl14', 'lunit-vits8', 'midnight12k'],
                        help='Patch encoder to use')
    parser.add_argument(
        '--patch_encoder_ckpt_path', type=str, default=None,
        help=(
            "Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors). "
            "This is only needed in offline environments (e.g., compute clusters without internet). "
            "If not provided, models are downloaded automatically from Hugging Face. "
            "You can also specify local paths via the model registry at "
            "`./trident/patch_encoder_models/local_ckpts.json`."
        )
    )
    parser.add_argument('--slide_encoder', type=str, default=None, 
                        choices=['threads', 'titan', 'prism', 'gigapath', 'chief', 'madeleine',
                                 'mean-virchow', 'mean-virchow2', 'mean-conch_v1', 'mean-conch_v15', 'mean-ctranspath',
                                 'mean-gigapath', 'mean-resnet50', 'mean-hoptimus0', 'mean-phikon', 'mean-phikon_v2',
                                 'mean-musk', 'mean-uni_v1', 'mean-uni_v2',  
                                 ], 
                        help='Slide encoder to use')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for feature extraction. Defaults to 32.')
    
    # SAM arguments
    parser.add_argument('--use_sam', action='store_true', 
                        help='Use SAM for mask generation during feature extraction. Defaults to False.')
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model architecture type. Defaults to vit_h.')
    parser.add_argument('--sam_checkpoint_path', type=str, default=None,
                        help='Path to SAM model checkpoint file. Required if use_sam is True.')
    parser.add_argument('--sam_model_cfg', type=str, default=None,
                        help='Path to SAM model configuration file. Required if use_sam is True and SAM version is sam2.')
    parser.add_argument('--sam_version', type=str, default='sam',
                        choices=['sam', 'sam2'],
                        help='SAM version to use. Defaults to sam.')
    parser.add_argument('--sam_pred_iou_thresh', type=float, default=0.3,
                        help='SAM prediction IoU threshold for mask filtering. Defaults to 0.3.')
    parser.add_argument('--sam_stability_score_thresh', type=float, default=0.6,
                        help='SAM stability score threshold for mask filtering. Defaults to 0.6.')
    parser.add_argument('--sam_min_mask_region_area', type=float, default=0.1,
                        help='SAM minimum mask area as a fraction of image area. Defaults to 0.1.')
    parser.add_argument('--sam_debug', type=str, default=None,
                        help='Directory name to save SAM segmentation visualizations for debugging. If None, no debug visualizations are saved. Defaults to None.')
    return parser


def parse_arguments():
    return build_parser().parse_args()


def generate_help_text() -> str:
    """
    Generate the command-line help text for documentation purposes.
    
    Returns:
        str: The full help message string from the argument parser.
    """
    parser = build_parser()
    return parser.format_help()


def initialize_processor(args):
    """
    Initialize the Trident Processor with the given arguments.
    """
    return Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        clear_cache=args.clear_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis,
        max_workers=args.max_workers,
        reader_type=args.reader_type
    )


def run_task(processor, args):
    """
    Execute the specified task using the Trident Processor.
    """
    if args.task == 'cache':
        processor.populate_cache()
    elif args.task == 'seg':
        # Minimal example for tissue segmentation:
        # python run_batch_of_slides.py --task seg --wsi_dir wsis --job_dir trident_processed --gpu 0
        from trident.segmentation_models.load import segmentation_model_factory

        # instantiate segmentation model and artifact remover if requested by user
        segmentation_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
            )
        else:
            artifact_remover_model = None

        # run segmentation 
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue= not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            device=f'cuda:{args.gpu}',
        )
    elif args.task == 'coords':
        # Minimal example for tissue patching:
        # python run_batch_of_slides.py --task coords --wsi_dir wsis --job_dir trident_processed --mag 20 --patch_size 256
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion
        )
    elif args.task == 'feat':
        if args.slide_encoder is None: # Run patch encoder:
            # Minimal example for feature extraction:
            # python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)
            
            # Prepare SAM configuration if use_sam is True
            sam_config = None
            if args.use_sam:
                if args.sam_checkpoint_path is None:
                    raise ValueError("SAM checkpoint path must be provided when use_sam is True")
                sam_config = {
                    "model_type": args.sam_model_type,
                    "checkpoint_path": args.sam_checkpoint_path,
                    "sam_version": args.sam_version,
                    "sam_model_cfg": args.sam_model_cfg,
                    "pred_iou_thresh": args.sam_pred_iou_thresh,
                    "stability_score_thresh": args.sam_stability_score_thresh,
                    "min_mask_region_area": args.sam_min_mask_region_area,
                    "debug": args.sam_debug
                }
            
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.batch_size,
                use_sam=args.use_sam,
                sam_config=sam_config
            )
        else:
            # Minimal example for feature extraction:
            # python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir trident_processed --slide_encoder threads --mag 20 --patch_size 256
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.batch_size
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def main():
    args = parse_arguments()
    processor = initialize_processor(args)

    # ensure cuda is available
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.task == 'all':
        args.task = 'seg'
        run_task(processor, args)
        args.task = 'coords'
        run_task(processor, args)
        args.task = 'feat'
        run_task(processor, args)
    else:
        run_task(processor, args)


if __name__ == "__main__":
    main()
