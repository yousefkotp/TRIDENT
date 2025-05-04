__version__ = "0.1.1"

from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.WSIFactory import load_wsi, WSIReaderType
from trident.wsi_objects.WSIPatcher import OpenSlideWSIPatcher, WSIPatcher
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset

from trident.Visualization import visualize_heatmap

from trident.Processor import Processor

from trident.Converter import AnyToTiffConverter

from trident.Maintenance import deprecated

__all__ = [
    "Processor",
    "load_wsi",
    "OpenSlideWSI", 
    "ImageWSI",
    "CuCIMWSI",
    "WSIPatcher",
    "OpenSlideWSIPatcher",
    "WSIPatcherDataset",
    "visualize_heatmap",
    "AnyToTiffConverter",
    "deprecated",
    "WSIReaderType",
]

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
        reader_type=args.reader_type,
        search_nested=args.search_nested,
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
            batch_size=args.seg_batch_size,
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
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size,
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
                batch_limit=args.feat_batch_size
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')
