import splitfolders  # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("G:/My Drive/College/Sem-6/BTP/ds_col", output="G:/My Drive/College/Sem-6/BTP/ds_col", seed=42, ratio=(.8, .2), group_prefix=None) # default values
splitfolders.ratio("G:/My Drive/College/Sem-6/BTP/ds_row", output="G:/My Drive/College/Sem-6/BTP/ds_row", seed=42, ratio=(.8, .2), group_prefix=None) 
#python -m keras_segmentation verify_dataset --images_path="G:/My Drive/College/Sem-6/BTP/ds_row/final_row/images_prepped_train/" --segs_path="G:/My Drive/College/Sem-6/BTP/ds_row/final_row/annotations_prepped_train/" --n_classes=50