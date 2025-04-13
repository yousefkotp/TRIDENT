Frequently Asked Questions
==========================

**Q:** How do I extract embeddings from legacy CLAM coordinates?
**A:**

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir legacy_dir --coords_dir extracted_coords --patch_encoder uni_v1

**Q:** My WSIs have no MPP metadata. What should I do?
**A:** Use the `--custom_list_of_wsis` flag with a CSV including `wsi` and `mpp` columns.

**Q:** I want to skip patches on holes.
**A:** Use `--remove_holes`. Default is to include all tissue regions.