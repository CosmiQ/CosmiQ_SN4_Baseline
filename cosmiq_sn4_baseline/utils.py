import os
import numpy as np
from spacenetutilities.labeltools import coreLabelTools as cLT


def create_mask(geojson_dir, src_im_dir, dest_im_dir, chip_id):
    """Create masks using source images and geojson for a given chip ID.

    Arguments:
    ----------
    geojson_dir (str): Path to geojson directory.
    src_im_dir (str): Path to source images, needed for geospatial info.
    dest_im_dir (str): Directory to save masks to.
    chip_id (str): Name of the image/geojson chip being masked.

    Returns:
    --------
    Saves a Tiff-formatted mask at path dest_im_dir/mask_[chip_id].tif.


    """
    try:
        src_geojson = [f for f in os.listdir(geojson_dir) if chip_id in f][0]
    except IndexError:
        print('WARNING: geojson not found for chip # {}'.format(chip_id))
        return
    try:
        src_im = [f for f in os.listdir(src_im_dir) if chip_id in f][0]
    except IndexError:
        print('WARNING: image not found for chip # {}'.format(chip_id))
        return
    src_geojson = os.path.join(geojson_dir, src_geojson)
    dest_im = os.path.join(dest_im_dir, 'mask_'+chip_id+'.tif')
    src_im = os.path.join(src_im_dir, src_im)
    cLT.createRasterFromGeoJson(src_geojson, src_im, dest_im)


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
