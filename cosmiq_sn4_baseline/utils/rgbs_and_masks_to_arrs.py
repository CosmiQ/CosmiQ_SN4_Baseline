import numpy as np
import cosmiq_sn4_baseline as space_base
import warnings
import os
import gc
from skimage import io

def rgbs_and_masks_to_arrs(rgb_src_dir, dest_path, mask_src_dir=None,
                           dataset_type='train', mk_angle_splits=True,
                           verbose=False, skip_existing=False):
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
    if dataset_type == 'train' and not os.path.isdir(mask_src_dir):
        raise NotADirectoryError('{} is not a directory'.format(mask_src_dir))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if dataset_type not in ['train', 'test']:
        raise ValueError('The only accepted options for `dataset_type` are ' +
                         ' ["train", "test"], got {}'.format(dataset_type))
    if dataset_type == 'train':
        make_training_arrs(rgb_src_dir, dest_path, mask_src_dir,
                           mk_angle_splits=mk_angle_splits,
                           verbose=verbose, skip_existing=skip_existing)

    elif dataset_type == 'test':
        make_test_arrs(rgb_src_dir, dest_path, mk_angle_splits=mk_angle_splits,
                       verbose=verbose, skip_existing=skip_existing)


def make_training_arrs(rgb_src_dir, dest_path, mask_src_dir,
                       train_val_split=0.8, mk_angle_splits=False,
                       verbose=False, skip_existing=False):
    """Make the training arrays."""
    rgb_im_list = [f for f in os.listdir(rgb_src_dir) if f.endswith('.tif')]
    mask_im_list = [f for f in os.listdir(mask_src_dir) if f.endswith('.tif')]
    n_ims = len(rgb_im_list)
    n_collects = len(space_base.COLLECTS)
    if n_ims != n_collects * len(mask_im_list):
        warnings.warn('There is a mismatch between the number of rgb images ' +
                      'and the number of masks. This will disrupt your ' +
                      'training dataset.')
    unique_chips = ['_'.join(m.split('_')[1:])[:-4] for m in mask_im_list]
    n_chips = len(unique_chips)
    train_output_dir = os.path.join(dest_path, 'train')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if skip_existing and os.path.exists(os.path.join(train_output_dir,
                                                     'all_train_masks.npy')):
        if verbose:
            print('Mask array exists and overwrite is off. ' +
                  'Loading existing array.')
        mask_arr = np.load(os.path.join(train_output_dir, 'all_train_masks.npy',
                                        mmap_mode='r'))
    else:
        if verbose:
            print('Making mask array...')
        mask_arr = _make_mask_arr(mask_src_dir, unique_chips)
        if verbose:
            print('Mask array complete.')
    if skip_existing and os.path.exists(os.path.join(train_output_dir,
                                                     'all_train_ims.npy')):
        if verbose:
            print('Training image array already exists and overwrite is off.' +
                  ' Loading existing array.')
        im_arr = np.load(os.path.join(train_output_dir, 'all_train_ims.npy'),
                         mmap_mode='r')
    else:
        train_im_ids = []
        if verbose:
            print('Making image array...')
        im_arr = np.empty(shape=(len(space_base.COLLECTS), len(unique_chips),
                                     900, 900, 3), dtype='float16')  # higher precision unnecessary
        for collect_idx in range(n_collects):
            if verbose:
                print('  Working on collect #{} of 27...'.format(collect_idx))
            for chip_idx in range(len(unique_chips)):
                if verbose:
                    print('    Reading chip #{} of {}'.format(chip_idx,
                                                              n_chips))
                im_fname = '_'.join(['Pan-Sharpen',
                                    space_base.COLLECTS[collect_idx],
                                    unique_chips[chip_idx]]) + '.tif'
                train_im_ids.append(im_fname)
                # normalize image to 0-1 range while loading into the array
                im_arr[collect_idx,
                       chip_idx,
                       :, :, :] = io.imread(os.path.join(rgb_src_dir,
                                                         im_fname))/255
    if verbose:
        print('Initial image preparation complete.')
        print('Splitting masks into training and validation sets:')
        print('  {} percent train'.format(train_val_split*100))
        print('  {} percent validation'.format((1-train_val_split)*100))
    train_inds = np.random.choice(np.arange(n_chips),
                                  size=int(n_chips*train_val_split),
                                  replace=False)
    val_inds = np.array([i for i in np.arange(n_chips)
                         if i not in train_inds.tolist()])
    if verbose:
        print('Saving chip indices corresponding to train and validation sets.')
        print("YOU'LL NEED THESE FILES IF YOU WANT TO REBUILD THE SAME SPLIT!")
    np.save(os.path.join(dest_path, 'training_chip_ids.npy'),
            np.array(unique_chips)[train_inds])
    np.save(os.path.join(dest_path, 'validation_chip_ids.npy'),
            np.array(unique_chips)[val_inds])
    if verbose:
        print('Splitting train data into training and validation...')
    train_im_arr = im_arr[:, train_inds, :, :, :]
    train_mask_arr = mask_arr[train_inds, :, :, :]
    if verbose:
        print('Saving training arrays...')
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
    if verbose:
        print('Saving training image array...')
    # flatten the collect axis during saving for ease of use in model
    np.save(os.path.join(train_output_dir,
                         'all_train_ims.npy'),
            np.concatenate([train_im_arr[i, :, :, :, :] for i in
                            range(n_collects)]))
    gc.collect()
    if mk_angle_splits:
        if verbose:
            print('  Saving sub-arrays for each subset of angles...')
            print('    Saving nadir training arrays...')
        nadir_arr_path = os.path.join(train_output_dir,
                                      'nadir_train_ims.npy')
        offnadir_arr_path = os.path.join(train_output_dir,
                                         'offnadir_train_ims.npy')
        faroffnadir_arr_path = os.path.join(train_output_dir,
                                            'faroffnadir_train_ims.npy')
        nadir_mask_path = os.path.join(train_output_dir,
                                       'nadir_train_masks.npy')
        offnadir_mask_path = os.path.join(train_output_dir,
                                          'offnadir_train_masks.npy')
        faroffnadir_mask_path = os.path.join(train_output_dir,
                                             'faroffnadir_train_masks.npy')
        if not skip_existing or not os.path.exists(nadir_arr_path):
            np.save(os.path.join(train_output_dir, 'nadir_train_ims.npy'),
                    np.concatenate([train_im_arr[i, :, :, :, :]
                                    for i in range(11)]))  # first 11 are nadir
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(nadir_arr_path))
        if not skip_existing or not os.path.exists(offnadir_arr_path):
            if verbose:
                print('    Saving off-nadir training arrays...')
            np.save(os.path.join(train_output_dir, 'offnadir_train_ims.npy'),
                    np.concatenate([train_im_arr[i, :, :, :, :]
                                    for i in range(11, 18)]))  # these are off-nadir
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(offnadir_arr_path))
        if not skip_existing or not os.path.exists(faroffnadir_arr_path):
            if verbose:
                print('    Saving far-off-nadir training arrays...')
            np.save(os.path.join(train_output_dir, 'faroffnadir_train_ims.npy'),
                    np.concatenate([train_im_arr[i, :, :, :, :]
                                    for i in range(18, 27)]))  # these are far-off
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(faroffnadir_arr_path))
        if verbose:
            print('    Saving masks for subsets...')
        if not skip_existing or not os.path.exists(nadir_mask_path):
            np.save(os.path.join(train_output_dir, 'nadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(11)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(nadir_mask_path))
        if not skip_existing or not os.path.exists(offnadir_mask_path):
            np.save(os.path.join(train_output_dir, 'offnadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(11, 18)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(offnadir_mask_path))
        if not skip_existing or not os.path.exists(faroffnadir_mask_path):
            np.save(os.path.join(train_output_dir, 'faroffnadir_train_masks.npy'),
                    np.concatenate([train_mask_arr for i in range(18, 27)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(faroffnadir_mask_path))

    if verbose:
        print('Cleaning up training arrays...')
    train_im_arr = None
    train_mask_arr = None
    gc.collect()
    if verbose:
        print('Getting there. Now preparing validation arrays...')
    val_im_arr = im_arr[:, val_inds, :, :, :]
    val_mask_arr = mask_arr[val_inds, :, :, :]
    val_output_dir = os.path.join(dest_path, 'validate')
    if not os.path.exists(val_output_dir):
        os.mkdir(val_output_dir)
        if verbose:
            print('Saving val image array...')
        # flatten the collect axis during saving for ease of use in model
        np.save(os.path.join(val_output_dir,
                             'all_val_ims.npy'),
                np.concatenate([val_im_arr[i, :, :, :, :] for i in
                                range(n_collects)]))
    if mk_angle_splits:
        if verbose:
            print('  Saving sub-arrays for each subset of angles...')
            print('    Saving nadir val arrays...')
        nadir_arr_path = os.path.join(val_output_dir,
                                      'nadir_val_ims.npy')
        offnadir_arr_path = os.path.join(val_output_dir,
                                         'offnadir_val_ims.npy')
        faroffnadir_arr_path = os.path.join(val_output_dir,
                                            'faroffnadir_val_ims.npy')
        nadir_mask_path = os.path.join(val_output_dir,
                                       'nadir_val_masks.npy')
        offnadir_mask_path = os.path.join(val_output_dir,
                                          'offnadir_val_masks.npy')
        faroffnadir_mask_path = os.path.join(val_output_dir,
                                             'faroffnadir_val_masks.npy')
        if not skip_existing or not os.path.exists(nadir_arr_path):
            np.save(os.path.join(val_output_dir, 'nadir_val_ims.npy'),
                    np.concatenate([val_im_arr[i, :, :, :, :]
                                    for i in range(11)]))  # first 11 are nadir
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(nadir_arr_path))
        if not skip_existing or not os.path.exists(offnadir_arr_path):
            if verbose:
                print('    Saving off-nadir val arrays...')
            np.save(os.path.join(val_output_dir, 'offnadir_val_ims.npy'),
                    np.concatenate([val_im_arr[i, :, :, :, :]
                                    for i in range(11, 18)]))  # these are off-nadir
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(offnadir_arr_path))
        if not skip_existing or not os.path.exists(faroffnadir_arr_path):
            if verbose:
                print('    Saving far-off-nadir val arrays...')
            np.save(os.path.join(val_output_dir, 'faroffnadir_val_ims.npy'),
                    np.concatenate([val_im_arr[i, :, :, :, :]
                                    for i in range(18, 27)]))  # these are far-off
        else:
            if verbose:
                print('Saved numpy array {} exists, skipping...'.format(faroffnadir_arr_path))
        if verbose:
            print('    Saving masks for subsets...')
        if not skip_existing or not os.path.exists(nadir_mask_path):
            np.save(os.path.join(val_output_dir, 'nadir_val_masks.npy'),
                    np.concatenate([val_mask_arr for i in range(11)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(nadir_mask_path))
        if not skip_existing or not os.path.exists(offnadir_mask_path):
            np.save(os.path.join(val_output_dir, 'offnadir_val_masks.npy'),
                    np.concatenate([val_mask_arr for i in range(11, 18)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(offnadir_mask_path))
        if not skip_existing or not os.path.exists(faroffnadir_mask_path):
            np.save(os.path.join(val_output_dir, 'faroffnadir_val_masks.npy'),
                    np.concatenate([val_mask_arr for i in range(18, 27)]))
        else:
            if verbose:
                print('     {} exists, skipping...'.format(faroffnadir_mask_path))


def make_test_arrs(rgb_src_dir, dest_path, mk_angle_splits=False, verbose=False,
                   skip_existing=False):
    """Make test arrays."""
    rgb_im_list = [f for f in os.listdir(rgb_src_dir) if f.endswith('.tif')]
    n_ims = len(rgb_im_list)
    output_dir = os.path.join(dest_path, 'test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if verbose:
        print('Making image array...')
    if skip_existing and os.path.exists(os.path.join(output_dir,
                                                     'all_test_ims.npy')):
        if verbose:
            print('Test image array already exists and overwrite is off.' +
                  ' Loading existing array.')
        im_arr = np.load(os.path.join(output_dir, 'all_test_ims.npy'),
                         mmap_mode='r')
        all_im_list = np.load(os.path.join(output_dir,
                                           'test_im_fnames.py')).tolist()
        angles = np.array(
            [int((f.split('_')[2].strip('nadir'))) for f in all_im_list]
            )
    else:
        all_im_list = [f for f in os.listdir(rgb_src_dir)
                       if f.endswith('.tif')]
        all_im_list.sort()
        angles = np.array(
            [int((f.split('_')[2].strip('nadir'))) for f in all_im_list]
            )
        im_arr = np.empty(shape=(angles.size, 900, 900, 3))
        if verbose:
            print('Making image array...')
        n_ims = len(all_im_list)
        for im_idx in range(n_ims):
            if verbose:
                print('    Reading test image #{} of {}'.format(im_idx, n_ims))
                # normalize image to 0-1 range while loading into the array
                im_arr[im_idx,
                       :, :, :] = io.imread(os.path.join(
                           rgb_src_dir, all_im_list[im_idx]))/255
        if verbose:
            print('Saving test image array...')
        np.save(os.path.join(output_dir, 'all_test_ims.npy'), im_arr)
        np.save(os.path.join(output_dir, 'test_im_fnames.npy'),
                np.array(all_im_list))
    if mk_angle_splits:
        if verbose:
            print('Saving angle subsets...')
        if not skip_existing or not os.path.exists(os.path.join(output_dir, 'nadir_test_ims.npy')):
            np.save(os.path.join(output_dir, 'nadir_test_ims.npy'),
                    im_arr[angles < 26, :, :, :])
            np.save(os.path.join(output_dir, 'nadir_im_fnames.npy'),
                    np.array(all_im_list)[angles < 26])
        if not skip_existing or not os.path.exists(os.path.join(output_dir, 'offnadir_test_ims.npy')):
            np.save(os.path.join(output_dir, 'offnadir_test_ims.npy'),
                    im_arr[np.logical_and(angles > 25, angles < 40), :, :, :])
            np.save(os.path.join(output_dir, 'offnadir_im_fnames.npy'),
                    np.array(all_im_list)[np.logical_and(angles > 25,
                                                         angles < 40)])
        if not skip_existing or not os.path.exists(os.path.join(output_dir, 'faroffnadir_test_ims.npy')):
            np.save(os.path.join(output_dir, 'faroffnadir_test_ims.npy'),
                    im_arr[angles > 40, :, :, :])
            np.save(os.path.join(output_dir, 'faroffnadir_im_fnames.npy'),
                    np.array(all_im_list)[angles > 40])


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
