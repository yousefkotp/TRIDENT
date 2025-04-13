Frequently Asked Questions
==========================

.. dropdown:: **How do I extract embeddings from legacy CLAM coordinates?**

   Use the `--coords_dir` flag to pass CLAM-style patch coordinates:

   .. code-block:: bash

      python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir legacy_dir --coords_dir extracted_coords --patch_encoder uni_v1


.. dropdown:: **My WSIs have no micron-per pixel (MPP) or magnigication metadata. What should I do?**

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

   By default, TRIDENT includes all tissue patches (including holes). Use `--remove_holes` to exclude them. No recommended, as "holes" are often helping defining the tissue microenvironment.
