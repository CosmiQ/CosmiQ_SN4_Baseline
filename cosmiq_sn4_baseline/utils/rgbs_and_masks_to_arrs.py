import numpy as np
import cosmiq_sn4_baseline as space_base
import warnings
import os
import gc
from skimage import io

def rgbs_and_masks_to_arrs(rgb_src_dir, dest_path, mask_src_dir=None,
                           train_val_split=1, mk_angle_splits=True,
                           verbose=False):
    """Convert RGB images to NumPy arrays for training and validation.

    Arguments:
    ---------
    im_src_dir (str): Path to directory containing RGB 8-bit images produced by
        `make_rgbs`.
    dest_path (str): Path to output directory to store NumPy array outputs.
    mask_src_dir (str): If generating mask array (training/validation data,
        path to directory containing mask images produced by
        `masks_from_geojsons`. Default `None` means no masks, only produce im
        array.
    train_val_split (float, range[0, 1]): What fraction of images should go to
        train-val subsets? Defaults to 1 (no splitting). Number corresponds to
        fraction that will go into the training set, 1-`train_val_split` will
        go to validation set.
    mk_angle_splits (bool): Create nadir, off-nadir, and far-off-nadir
        sub-arrays? Defaults to yes (True). Note that this doubles the amount
        of hard disk space required to store the dataset.
    verbose (bool): Verbose text output. Defaults to False.

    NOTE: THIS FUNCTION MAKES SEVERAL VERY LARGE (>10 GB) NUMPY ARRAYS AND CAN
    TAKE A LONG TIME TO RUN, PARTICULARLY IF YOU DON'T HAVE A LOT OF MEMORY.
    SORRY.

    """
    if not os.path.isdir(rgb_src_dir):
        raise NotADirectoryError('{} is not a directory'.format(rgb_src_dir))
    if mask_src_dir is not None and not os.path.isdir(mask_src_dir):
        raise NotADirectoryError('{} is not a directory'.format(mask_src_dir))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    rgb_im_list = [f for f in os.listdir(rgb_src_dir) if f.endswith('.tif')]
    if mask_src_dir is not None:
        mask_im_list = [f for f in os.listdir(mask_src_dir) if f.endswith('.tif')]
    n_ims = len(rgb_im_list)
    # check to make sure src img count = number collects * number masks
    n_collects = len(space_base.COLLECTS)
    if mask_src_dir is not None and n_ims != n_collects * len(mask_im_list):
        warnings.warn('There is a mismatch between the number of rgb images and the number of masks. This will disrupt your training dataset.')
    unique_chips = ['_'.join(m.split('_')[1:])[:-4] for m in mask_im_list]
    n_chips = len(unique_chips)
    im_arr = np.empty(shape=(n_collects, n_chips, 900, 900, 3),
                      dtype='float16')
    if verbose and mask_src_dir is not None:
        print('Making mask array...')
    if mask_src_dir is not None:
        mask_arr = _make_mask_arr(mask_src_dir, unique_chips)
    if verbose and mask_src_dir is not None:
        print('Mask array complete.')
    if verbose:
        print('Making image array...')
    for collect_idx in range(n_collects):
        if verbose:
            print('  Working on collect #{} of 27...'.format(collect_idx))
        for chip_idx in range(len(unique_chips)):
            if verbose:
                print('    Reading chip #{} of {}'.format(chip_idx, n_chips))
            im_fname = '_'.join('Pan-Sharpen',
                                space_base.COLLECTS[collect_idx],
                                unique_chips[chip_idx]) + '.tif'
            # normalize image to 0-1 range while loading into the array
            im_arr[collect_idx, chip_idx, :, :, :] = io.imread(im_fname)/255
    if verbose:
        print('Initial image preparation complete.')
    if train_val_split != 1:
        print('Splitting masks into training and validation sets:')
        print('  {} percent train'.format(train_val_split*100))
        print('  {} percent validation'.format((1-train_val_split)*100))
    train_inds = np.random.choice(np.arange(n_chips),
                                  size=int(n_chips*train_val_split),
                                  replace=False)
    if train_val_split != 1:
        val_inds = np.array([i for i in np.arange(n_chips)
                             if i not in train_inds.tolist()])
    if verbose:
        print('Saving chip indices corresponding to train and validation sets.')
        print("YOU'LL NEED THESE FILES IF YOU WANT TO REBUILD THE SAME SPLIT!")
    if train_val_split != 1:
        np.save(os.path.join(dest_path, 'training_chip_ids.npy'),
                unique_chips[train_inds])
        np.save(os.path.join(dest_path, 'validation_chip_ids.npy'),
                unique_chips[val_inds])
    else:
        np.save(os.path.join(dest_path, 'test_chip_ids.npy'))
    if verbose and train_val_split != 1:
        print('Splitting train data into training and validation...')
    if train_val_split != 1:
        train_im_arr = im_arr[:, train_inds, :, :, :]
        train_mask_arr = mask_arr[train_inds, :, :, :]
    else:
        train_im_arr = im_arr
        if mask_src_dir is not None:
            train_mask_arr = mask_arr
    if train_val_split != 1:
        train_output_dir = os.path.join(dest_path, 'train')
    else:
        train_output_dir = os.path.join(dest_path, 'test')
    if verbose:
        print('Saving training arrays...')
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
        if verbose:
            print('Saving training image array...')
        # flatten the collect axis during saving for ease of use in model
        if train_val_split != 1:
            np.save(os.path.join(train_output_dir, 'all_train_ims.npy'),
                    np.concatenate([train_im_arr[i, :, :, :, :] for i in
                                    range(n_collects)]))
        else:
            np.save(os.path.join(train_output_dir, 'all_test_ims.npy'),
                    np.concatenate([train_im_arr[i, :, :, :, :] for i in
                                    range(n_collects)]))
        if verbose and mask_src_dir is not None:
            print('Saving training mask array...')
        # replicate the mask array to match collects in the image array
        if mask_src_dir is not None:
            np.save(os.path.join(train_output_dir, 'all_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(n_collects)]))
    if mk_angle_splits:
        if verbose:
            print('  Saving sub-arrays for each subset of angles...')
            print('    Saving nadir training arrays...')
        np.save(os.path.join(train_output_dir, 'nadir_train_ims.npy'),
                np.concatenate([train_im_arr[i, :, :, :, :]
                                for i in range(11)]))  # first 11 are nadir
        if verbose:
            print('    Saving off-nadir training arrays...')
        np.save(os.path.join(train_output_dir, 'offnadir_train_ims.npy'),
                np.concatenate([train_im_arr[i, :, :, :, :]
                                for i in range(11, 18)]))  # these are off-nadir
        if verbose:
            print('    Saving far-off-nadir training arrays...')
        np.save(os.path.join(train_output_dir, 'faroffnadir_train_ims.npy'),
                np.concatenate([train_im_arr[i, :, :, :, :]
                                for i in range(18, 27)]))  # these are far-off
        if mask_src_dir is not None:
            if verbose:
                print('    Saving masks for subsets...')
            np.save(os.path.join(train_output_dir, 'nadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(11)]))
            np.save(os.path.join(train_output_dir, 'faroffnadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(18, 27)]))
            np.save(os.path.join(train_output_dir, 'offnadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(11, 18)]))
    if verbose and train_val_split != 1:
        print('Cleaning up training arrays...')
    train_im_arr = None
    train_mask_arr = None
    gc.collect()
    if train_val_split == 1:
        print('Finished. Cleaning up...')
        im_arr = None
        mask_arr = None
        gc.collect()
        return
    if verbose:
        print('Getting there. Now preparing validation arrays...')
    val_im_arr = im_arr[:, val_inds, :, :, :]
    val_mask_arr = mask_arr[val_inds, :, :, :]
    val_output_dir = os.path.join(dest_path, 'validate')
    if not os.path.exists(val_output_dir):
        os.mkdir(val_output_dir)
        # flatten the collect axis during saving for ease of use in model
        if verbose:
            print('Saving validation image array...')
        np.save(os.path.join(val_output_dir, 'all_val_ims.npy'),
                np.concatenate([val_im_arr[i, :, :, :, :] for i in
                                range(n_collects)]))
        if verbose:
            print('Saving validation mask array...')
        # replicate the mask array to match collects in the image array
        np.save(os.path.join(val_output_dir, 'all_val_masks.npy'),
                np.concatenate([val_mask_arr for i in range(n_collects)]))

    if mk_angle_splits:
        if verbose:
            print('  Saving sub-arrays for each subset of angles...')
            print('    Saving nadir validation arrays...')
        np.save(os.path.join(val_output_dir, 'nadir_val_ims.npy'),
                np.concatenate([val_im_arr[i, :, :, :, :]
                                for i in range(11)]))  # first 11 are nadir

        np.save(os.path.join(val_output_dir, 'nadir_val_masks.npy'),
                np.concatenate([val_mask_arr for i in range(11)]))
        if verbose:
            print('    Saving off-nadir validation arrays...')
        np.save(os.path.join(val_output_dir, 'offnadir_val_ims.npy'),
                np.concatenate([val_im_arr[i, :, :, :, :]
                                for i in range(11, 18)]))  # these are off-nadir
        np.save(os.path.join(val_output_dir, 'offnadir_val_masks.npy'),
                np.concatenate([val_mask_arr for i in range(11, 18)]))
        if verbose:
            print('    Saving far-off-nadir validation arrays...')
        np.save(os.path.join(val_output_dir, 'faroffnadir_val_ims.npy'),
                np.concatenate([val_im_arr[i, :, :, :, :]
                                for i in range(18, 27)]))  # these are far-off
        np.save(os.path.join(val_output_dir, 'faroffnadir_val_masks.npy'),
                np.concatenate([val_mask_arr for i in range(18, 27)]))
    if verbose:
        print('YOU MADE IT! Cleaning up...')
    im_arr = None
    mask_arr = None
    val_im_arr = None
    val_mask_arr = None
    gc.collect()
    if verbose:
        print("Done with cleanup. You're ready to train your model!")


def _make_mask_arr(mask_src_dir, unique_chips):
    """Make a numpy array of masks."""
    masks = [f for f in os.listdir(mask_src_dir) if f.endswith('tif')]
    mask_out_arr = np.empty(shape=(len(unique_chips), 900, 900, 1),
                            dtype='bool')
    for idx in range(len(unique_chips)):
        mask_fname = [f for f in masks if unique_chips[idx] in f][0]  # sorry
        mask = io.imread(os.path.join(mask_src_dir, mask_fname)) > 0
        mask_out_arr[idx, :, :, :] = mask[:, :, np.newaxis]
    return mask_out_arr
