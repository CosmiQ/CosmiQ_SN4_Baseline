import numpy as np
import rasterio
import os
import cosmiq_sn4_baseline as space_base
import cv2


def convert_to_8bit_bgr(four_channel_im, threshold):
    """Produce a three-channel, 8-bit RGB from four-channel, 16-bit.

    Arguments:
    ----------
    four_channel_im (uint16 numpy array of shape [y, x, 4]): Source image to
        convert to a 3-channel BGR output. Loading 4-channel pan-sharpened
        geotiffs yield this input format.
    threshold (int): Threshold to cut off input array at.

    Returns:
    --------
    A [y, x, 3]-shaped 8-bit numpy array with channel order B-G-R.

    This function assumes channel order is BGRX, where X will be discarded.
    """
    three_channel_im = four_channel_im[:, :, 0:3]  # remove 4th channel
    # next, clip to threshold
    np.clip(three_channel_im, None, threshold, out=three_channel_im)
    # finally, rescale to 8-bit range with threshold value scaled to 255
    three_channel_im = np.floor_divide(three_channel_im,
                                       threshold/255).astype('uint8')
    return three_channel_im


def pan_to_bgr(src_path, dest_path):
    """Load a pan-sharpened BGRX image and output a BGR 8-bit numpy array.

    Arguments:
    ---------
    src_path (str): Full path to the pan-sharpened source image.
    dest_path (str): Full path to the destination BGR 8-bit file.

    """
    if os.path.exists(dest_path):  # if the file already exists - allows
        return
    im_reader = rasterio.open(os.path.join(src_path))
    img = np.empty((im_reader.height,
                    im_reader.width,
                    im_reader.count))
    for band in range(im_reader.count):
        img[:, :, band] = im_reader.read(band+1)
    bgr_im = convert_to_8bit_bgr(img, space_base.BGR_8BIT_THRESHOLD)
    cv2.imwrite(dest_path, bgr_im)


def make_rgbs(src_dir, dest_dir, verbose=False, skip_existing=True):
    """Create RGB images from Pan-Sharpened 16-bit source images.

    Arguments:
    ---------
    src_dir (str): Path to the source dir containing imagery. This can be
        either the SpaceNet-Off-Nadir_Train directory path (use `has_subdirs`)
        or the SpaceNet-Off-Nadir_Test directory path (`has_subdirs=False`).
    dest_dir (str): Path to the directory to save RGB-formatted imagery into.
        No subdirs will be created; flat structure with filenames IDing
        images from different collects.
    verbose (bool): Print verbose text output? Defaults to False.
    skip_existing (bool): Should source images be skipped if the destination
        file already exists? Defaults to True. If set to False, will overwrite
        existing RGB images with the same filenames in the destination dir.

    """
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    for collect in space_base.COLLECTS:
        if verbose:
            print('Converting collect {} to BGR 8-bit'.format(collect))
        collect_path = os.path.join(src_dir, collect)
        collect_pansharp_path = os.path.join(collect_path, 'Pan-Sharpen')
        im_list = [f for f in os.listdir(collect_pansharp_path)
                   if f.endswith('.tif')]
        n_ims = len(im_list)
        for i in range(n_ims):
            im_fname = im_list[i]
            dest_path = os.path.join(dest_dir, im_fname)
            if os.path.exists(dest_path) and skip_existing:
                if verbose:
                    print('8-bit RGB {} exists, skipping...'.format(dest_path))
                continue
            pan_to_bgr(os.path.join(collect_pansharp_path, im_fname),
                       os.path.join(dest_path))
            if verbose:
                print('    image {} of {} done'.format(i, n_ims))
