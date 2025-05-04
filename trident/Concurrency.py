import os
import copy
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from trident import Processor, initialize_processor, run_task


def cache_batch(wsis, batch_id, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for wsi in wsis:
        shutil.copy(wsi, os.path.join(dest_dir, os.path.basename(wsi)))
        if wsi.endswith(('.mrxs')):
            mrxs_dir = os.path.splitext(wsi)[0]
            if os.path.exists(mrxs_dir):
                shutil.copytree(mrxs_dir, os.path.join(dest_dir, os.path.basename(mrxs_dir)))

def get_all_valid_slides(args):
    if args.custom_list_of_wsis is not None:
        import pandas as pd
        wsi_df = pd.read_csv(args.custom_list_of_wsis)
        if 'wsi' not in wsi_df.columns:
            raise ValueError("CSV with custom list of WSIs must contain a column named 'wsi'.")
        if len(wsi_df['wsi'].dropna()) == 0:
            raise ValueError("No valid slides found in the custom list.")
        valid_slides = wsi_df['wsi'].dropna().astype(str).tolist()
        
        def exists_fn(rel_path):
            return os.path.exists(os.path.join(args.wsi_dir, rel_path))

        with ThreadPoolExecutor(max_workers=args.max_workers) as exe:
            # executor.map returns results in order
            for rel_path, ok in zip(valid_slides, exe.map(exists_fn, valid_slides)):
                if not ok:
                    raise ValueError(f"Slide {rel_path} not found in {args.wsi_dir}. If the folder is nested, you should set wsi column using the relative path to the wsi_source.")
    else:
        from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS
        wsi_ext = args.wsi_ext or (list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS))
        if args.search_nested:
            for root, _, files in os.walk(args.wsi_dir):
                for f in files:
                    if any(f.endswith(ext) for ext in wsi_ext):
                        rel_path = os.path.relpath(os.path.join(root, f), args.wsi_dir)
                        valid_slides.append(rel_path)
        else:
            valid_slides = [f for f in os.listdir(args.wsi_dir) if any(f.endswith(ext) for ext in wsi_ext)]
        valid_slides = sorted(valid_slides)

    valid_slides = [os.path.join(args.wsi_dir, slide) for slide in valid_slides]
    return valid_slides

def batch_producer(queue, valid_slides, start_idx, args):
    for i in range(start_idx, len(valid_slides), args.cache_batch_size):
        batch_paths = valid_slides[i:i + args.cache_batch_size]
        batch_id = i // args.cache_batch_size
        ssd_batch_dir = os.path.join(args.wsi_cache, f"batch_{batch_id}")
        print('Caching batch:', ssd_batch_dir)
        cache_batch(batch_paths, batch_id, ssd_batch_dir)
        queue.put(batch_id)  # Will block if SSD is full
    queue.put(None)  # Sentinel


def batch_consumer(queue, args):
    while True:
        batch_id = queue.get()
        if batch_id is None:
            queue.task_done()
            break
        ssd_batch_dir = os.path.join(args.wsi_cache, f"batch_{batch_id}")
        consumer_args = copy.deepcopy(args)
        
        consumer_args.wsi_dir = ssd_batch_dir  # temporarily override to point to SSD
        consumer_args.wsi_cache = None
        consumer_args.custom_list_of_wsis = None
        consumer_args.search_nested = False
        processor = initialize_processor(consumer_args)
        print('Processing batch:', ssd_batch_dir)
        
        if consumer_args.task == 'all':
            for task in ['seg', 'coords', 'feat']:
                consumer_args.task = task
                run_task(processor, consumer_args)
        else:
            processor = initialize_processor(consumer_args)
            run_task(processor, consumer_args)

        if consumer_args.clear_cache:
            shutil.rmtree(ssd_batch_dir)
        
        del processor
        queue.task_done()