# STEPS THAT MUST BE COMPLETED PRIOR TO RUNNING THIS SCRIPT:
#
# 1. Create and start the Docker container using the Dockerfile
# 2. Download the Off-Nadir train and test datasets from the S3 container using
#    the AWS CLI (see the challenge problem statement from TopCoder for
#    instructions)
# 3. Expand the .gzipped directories of images

import cv2
import os
import sys
from skimage import io
import numpy as np
import pandas as pd
import rasterio
import argparse
import cosmiq_sn4_baseline as space_base


def _check_data(dataset_dir):
    """Check to make sure dataset is structured correctly."""
    train_src_dir = os.path.join(dataset_dir,
                                 'SpaceNet-Off-Nadir_Train')
    test_src_dir = os.path.join(dataset_dir,
                                'SpaceNet-Off-Nadir_Test',
                                'SpaceNet-Off-Nadir_Test_Public')
    geojson_dir = os.path.join(train_src_dir, 'geojson')

    if not os.path.isdir(train_src_dir):
        raise NotADirectoryError('The source directory {} does not exist. See script for instructions.'.format(train_src_dir))
    if not os.path.isdir(test_src_dir):
        raise NotADirectoryError('The source directory {} does not exist. See script for instructions.'.format(test_src_dir))
    if not os.path.isdir(geojson_dir):
        raise NotADirectoryError('The GeoJSON tarball needs to be unzipped to proceed.')
    for collect in space_base.COLLECTS:
        collect_path = os.path.join(train_src_dir, collect)
        if not os.path.isdir(collect_path):
            raise NotADirectoryError('The imagery tarballs must be unzipped to proceed.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', '-d', type=str, required=True,
        help='Path to the directory containing unzipped imagery folders.'
        )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='sn4_processed',
        help='Path to the desired destination directory.'
        )
    parser.add_argument(
        '--verbose', '-v', action='store_const', const=True, default=False,
        help='Print verbose output while running.'
        )
    parser.add_argument(
        '--create_splits', '-s', action='store_const', const=True,
        default=False, help='Create nadir, off-nadir, and far-off-nadir subarrays. Takes up more hard disk space.'
    )
    args = parser.parse_args()
    if args.verbose:
        print('------------------------------------------------------------')
        print('                 Checking directories... ')
        print('------------------------------------------------------------')
    _check_data(args.dataset_dir)

    train_rgb_dir = os.path.join(args.output_dir, 'train_rgb')
    test_rgb_dir = os.path.join(args.output_dir, 'test_rgb')
    train_mask_dir = os.path.join(args.output_dir, 'masks')
    train_src_dir = os.path.join(args.dataset_dir, 'SpaceNet-Off-Nadir_Train')
    test_src_dir = os.path.join(args.dataset_dir, 'SpaceNet-Off-Nadir_Test',
                                'SpaceNet-Off-Nadir_Test_Public')
    geojson_src_dir = os.path.join(train_src_dir, 'geojson', 'spacenet-buildings')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(train_rgb_dir):
        os.mkdir(train_rgb_dir)
    if not os.path.exists(test_rgb_dir):
        os.mkdir(test_rgb_dir)
    if not os.path.exists(train_mask_dir):
        os.mkdir(train_mask_dir)

    if args.verbose:
        print('------------------------------------------------------------')
        print('         Making 8-bit RGB images for training. ')
        print('------------------------------------------------------------')
    space_base.utils.make_rgbs(train_src_dir, train_rgb_dir,
                               verbose=args.verbose)
    if args.verbose:
        print('------------------------------------------------------------')
        print('         Making 8-bit RGB images for testing. ')
        print('------------------------------------------------------------')
    space_base.utils.make_rgbs(test_src_dir, test_rgb_dir,
                               verbose=args.verbose)
    if args.verbose:
        print('------------------------------------------------------------')
        print('             Making masks from geojsons. ')
        print('------------------------------------------------------------')
    space_base.utils.masks_from_geojsons(geojson_src_dir,
                                         os.path.join(train_src_dir,
                                                      space_base.COLLECTS[0],
                                                      'Pan-Sharpen'),
                                         train_mask_dir, skip_existing=True,
                                         verbose=args.verbose)
    if args.verbose:
        print('------------------------------------------------------------')
        print('Creating train and val numpy arrays. This will take a while.')
        print('------------------------------------------------------------')
    space_base.utils.rgbs_and_masks_to_arrs(
        train_rgb_dir, args.output_dir, train_mask_dir,
        train_val_split=0.8, mk_angle_splits=args.create_splits,
        verbose=args.verbose)
    if args.verbose:
        print('------------------------------------------------------------')
        print('     Creating test numpy arrays. This will be faster. ')
        print('------------------------------------------------------------')
    space_base.utils.rgbs_and_masks_to_arrs(
        test_rgb_dir, os.path.join(args.output_dir, 'test_data'),
        verbose=args.verbose
        )
    if args.verbose:
        print('------------------------------------------------------------')
        print('  Transferring geotiffs to test dir for making predictions. ')
        print('------------------------------------------------------------')
    geotiff_dest_path = os.path.join(args.output_dir, 'test_data', 'geotiffs')
    if not os.path.exists(geotiff_dest_path):
        os.mkdir(geotiff_dest_path)
    for collect in space_base.COLLECTS:
        src_subdir = os.path.join(args.test_src_dir, collect, 'Pan-Sharpen')
        im_list = [f for f in os.listdir(src_subdir) if f.endswith('.tif')]
        for im_file in im_list:
            os.rename(os.path.join(src_subdir, im_file),
                      os.path.join(geotiff_dest_path, im_file))


if __name__ == '__main__':
    main()
