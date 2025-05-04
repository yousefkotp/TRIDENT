"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""
import os
import argparse
import torch
from trident import Processor, WSIReaderType, initialize_processor, run_task


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
    parser.add_argument('--cache_batch_size', type=int, default=32,
                        help='Number of slides to cache at once. This is to avoid filling up the local disk.')
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
    parser.add_argument("--search_nested", action="store_true",
                        help=("If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
                              "`wsi_source`. Uses `os.walk` to include slides from nested folders. "
                              "This allows processing of datasets organized in hierarchical structures. "
                              "Defaults to False (only top-level slides are included)."))
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
    parser.add_argument('--seg_batch_size', type=int, default=64, 
                        help='Batch size for segmentation. Defaults to 64.')
    parser.add_argument('--feat_batch_size', type=int, default=32, 
                        help='Batch size for feature extraction. Defaults to 32.')
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


def main():
    args = parse_arguments()
    # ensure cuda is available
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.wsi_cache and args.task != 'cache' and args.clear_cache:
        from queue import Queue
        from threading import Thread
        from trident.Concurrency import batch_producer, batch_consumer, get_all_valid_slides, cache_batch
        queue = Queue(maxsize=1)
        valid_slides = get_all_valid_slides(args)
        # copy first batch
        warm = valid_slides[:args.cache_batch_size]
        print('Warmup caching batch:', os.path.join(args.wsi_cache, "batch_0"))
        cache_batch(warm, 0, os.path.join(args.wsi_cache, "batch_0"))
        queue.put(0)
        print('Cache for first batch done. Starting processing.')
        start_idx = args.cache_batch_size

        producer = Thread(target=batch_producer, args=(queue, valid_slides, start_idx, args))
        consumer = Thread(target=batch_consumer, args=(queue, args))

        producer.start()
        consumer.start()

        producer.join()
        consumer.join()
    else:
        processor = initialize_processor(args)
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
