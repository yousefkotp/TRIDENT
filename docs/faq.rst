Frequently Asked Questions
==========================

.. dropdown:: **How do I extract embeddings from legacy CLAM coordinates?**

   Use the `--coords_dir` flag to pass CLAM-style patch coordinates:

   .. code-block:: bash

      python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir legacy_dir --coords_dir extracted_coords --patch_encoder uni_v1


.. dropdown:: **My WSIs have no micron-per-pixel (MPP) or magnification metadata. What should I do?**

   PNGs and JPEGs do not store MPP metadata in the file itself. If you're working with such formats, passing a CSV via `--custom_list_of_wsis` is **required**. This CSV should include at least two columns: `wsi` and `mpp`.

   Example:

   .. code-block:: csv

      wsi,mpp
      TCGA-AJ-A8CV-01Z-00-DX1_1.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_2.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_3.png,0.25

   If you're using OpenSlide-readable formats (e.g., `.svs`, `.tiff`), this CSV is optional—but you can still use it to:

   - Restrict processing to a specific subset of slides
   - Override incorrect or missing MPP metadata


.. dropdown:: **I want to skip patches on holes.**

   By default, TRIDENT includes all tissue patches (including holes). Use `--remove_holes` to exclude them. Not recommended, as "holes" often help define the tissue microenvironment.


.. dropdown:: **I work on a cluster without Internet access. How can I use models offline?**

   You can use local checkpoint files by editing the model registry files in Trident. This allows you to cache or pre-download all necessary models for both segmentation and patch encoding.

   **1. Segmentation Models**

   Update the segmentation model registry at:
   `trident/segmentation_models/local_ckpts.json`

   Example:

   .. code-block:: json

      {
        "hest": "./ckpts/trident/deeplabv3_seg_v4.ckpt",
        "grandqc": "./ckpts/trident/Tissue_Detection_MPP10.pth",
        "grandqc_artifact": "./ckpts/trident/GrandQC_MPP1_state_dict.pth"
      }

   **2. Patch Encoder Models**

   Update the patch encoder model registry at:
   `trident/patch_encoder_models/local_ckpts.json`

   Example:

   .. code-block:: json

      {
        "conch_v1": "./ckpts/conch_patch_encoder/pytorch_model.bin",
        "uni_v1": "./ckpts/uni_patch_encoder/pytorch_model.bin",
        "uni_v2": "./ckpts/uni2_patch_encoder/pytorch_model.bin",
        "ctranspath": "./ckpts/ctranspath_patch_encoder/CHIEF_CTransPath.pth",
        "phikon": "./ckpts/phikon_patch_encoder/pytorch_model.bin",
        "resnet50": "./ckpts/resnet_patch_encoder/pytorch_model.bin",
        "gigapath": "./ckpts/gigapath_patch_encoder/pytorch_model.bin",
        "virchow": "./ckpts/virchow_patch_encoder/pytorch_model.bin",
        "virchow2": "./ckpts/virchow2_patch_encoder/pytorch_model.bin",
        "hoptimus0": "./ckpts/hoptimus0_patch_encoder/pytorch_model.bin",
        "hoptimus1": "./ckpts/hoptimus1_patch_encoder/pytorch_model.bin",
        "phikon_v2": "./ckpts/phikon-v2_patch_encoder/model.safetensors",
        "kaiko-vitb8": "./ckpts/kaiko_vitb8_patch_encoder/model.safetensors",
        "kaiko-vitb16": "./ckpts/kaiko_vitb16_patch_encoder/model.safetensors",
        "kaiko-vits8": "./ckpts/kaiko_vits8_patch_encoder/model.safetensors",
        "kaiko-vits16": "./ckpts/kaiko_vits16_patch_encoder/model.safetensors",
        "kaiko-vitl14": "./ckpts/kaiko_vitl14_patch_encoder/model.safetensors",
        "lunit-vits8": "./ckpts/lunit_patch_encoder/model.safetensors",
        "conch_v15": "./ckpts/conchv1_5_patch_encoder/pytorch_model_vision.bin"
      }

   **3. Alternative Option**

   You can also directly pass a local checkpoint path at runtime using the `--patch_encoder_ckpt_path` argument in `run_batch_of_slides.py`.
