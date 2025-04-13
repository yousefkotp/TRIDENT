Quickstart
==========

🚀 Process a full directory of WSIs:

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256

🧪 Test a single WSI:

.. code-block:: bash

   python run_single_slide.py --slide_path ./wsis/sample.svs --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256

👣 Or go step-by-step:

- Tissue segmentation
- Patch extraction
- Feature extraction

Each step is documented in detail under the "Features" section.