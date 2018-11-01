import os
import numpy as np
import spacenetutilities.labeltools.coreLabelTools as cLT
import cosmiq_sn4_baseline as space_base
import argparse
import rasterio
from keras.models import load_model
import warnings
from cosmiq_sn4_baseline.inference import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', '-m', type=str,
        help='Path to the trained .hdf5 model file to use to generate predictions.'
        )
    parser.add_argument(
        '--test_dataset_path', '-t', type=str, required=True,
        help='Path to all_test_ims.npy produced by make_np_arrays.py.'
        )
    parser.add_argument(
        '--test_fnames_path', '-f', type=str, required=True,
        help='Path to [all/nadir/offnadir/faroffnadir]_im_fnames.npy produced by make_np_arrays.py.'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='test_output',
        help='Directory to output inference csv and geojsons.'
        )
    parser.add_argument(
        '--verbose', '-v', action='store_const', const=True, default=False,
        help='Verbose text output.'
        )
    parser.add_argument(
        '--angle_set', '-as', type=str, default='',
        help='Set of angles to run inference on. Can be left blank, or can ' +
             'provide `all`, `nadir`, `offnadir`, or `faroffnadir`.'
        )
    parser.add_argument(
        '--angle', '-a', type=int, nargs='+', default=0,
        help='Specific angle[s] to produce predictions for. Leave blank if ' +
             'using `--angle_set` option or to run on all angles.'
        )
    parser.add_argument(
        '--n_chips', '-n', type=int, default=0,
        help='Number of chips to test. Defaults to 0 (all). ' +
             '!!IMPORTANT NOTE: THIS WILL SUBSET ANGLES UNLESS USING `-r`!!'
        )
    parser.add_argument(
        '--randomize_chips', '-r', action='store_const',
        const=True, default=False,
        help='Randomize chip order prior to inference. Only matters if using' +
             '`--n_chips` to run on a subset.'
        )
    parser.add_argument(
        '--footprint_threshold', '-ft', type=int, default=0,
        help='Minimum footprint size in square pixels to save for output. ' +
             'All footprints smaller than the provided cutoff will be ' +
             'discarded.'
        )
    parser.add_argument(
        '--window_step', '-ws', type=int, default=64,
        help='Step size for sliding window during inference. Window will ' +
             'step this far in x and y directions, and all inferences will ' +
             'be averaged. Helps avoid edge effects. Defaults to 64 pxs. ' +
             'to have no overlap between windows, use the size of the ' +
             'window being tested (512 for ternausnetv1 and unet defaults)'
    )
    args = parser.parse_args()

    if args.verbose:
        print('--------------------------------------------------------------')
        print('                 Beginning inference...')
        print('--------------------------------------------------------------')
        print()
        print('Loading model {}...'.format(args.model_path))
    model = load_model(args.model_path, custom_objects={
        'hybrid_bce_jaccard': space_base.losses.hybrid_bce_jaccard,
        'precision': space_base.metrics.precision,
        'recall': space_base.metrics.recall})
    input_shape = model.layers[0].input_shape[1:]  # need this for infer()

    if args.verbose:
        print('Loading test dataset...')
    test_dataset = np.load(args.test_dataset_path, mmap_mode='r')
    test_fnames = np.load(args.test_fnames_path)

    # subset test angles
    test_angle_mask = np.array([True for i in range(test_dataset.shape[0])])
    if args.angle_set:
        if args.angle_set == 'nadir':
            test_angle_mask = space_base.COLLECT_ANGLES < 26
        elif args.angle_set == 'offnadir':
            test_angle_mask = np.logical_and(space_base.COLLECT_ANGLES > 25,
                                             space_base.COLLECT_ANGLES < 40)
        elif args.angle_set == 'faroffnadir':
            test_angle_mask = space_base.COLLECT_ANGLES > 40
    elif args.angle:
        test_angle_mask = np.isin(space_base.COLLECT_ANGLES, args.angle)
    if not np.all(test_angle_mask):
        test_angles = space_base.COLLECT_ANGLES[test_angle_mask]
        test_collect_names = space_base.COLLECTS[test_angle_mask]
        # subset the dataset based on the collects included through the logic
        # above
        im_fname_mask = np.array([str(f) for f in test_fnames
                                  if str(f) in test_collect_names])
        # handle edge case of im_fname_mask being all False
        if not im_fname_mask.any():
            raise IndexError(
                'There were no images selected by the collect angle logic.'
                )
        test_dataset = test_dataset[im_fname_mask, :, :, :]
        # handle edge case of im_fname_mask only having one true value
        if len(test_dataset.shape) == 3:
            test_dataset = test_dataset[np.newaxis, :, :, :]

    # subset test chips
    if args.randomize_chips:
        # get shuffled ax order
        ax_shuffle = np.random.shuffle(np.arange(test_dataset.shape[0]))
        test_dataset = test_dataset[ax_shuffle, :, :, :]
        test_fnames = test_fnames[ax_shuffle]
    if args.n_chips:
        # NOTE: IF NOT RANDOMLY SHUFFLING, THIS WILL SUBSET ANGLES!
        test_dataset = test_dataset[:args.n_chips, :, :, :]
        test_fnames = test_fnames[:args.n_chips]

    # perform inference for each image in overlapping steps
    preds_arr = np.empty(test_dataset.shape[0:-1])
    for idx in range(test_dataset.shape[0]):
        if args.verbose:
            print('Processing image {} of {}'.format(idx,
                                                     test_dataset.shape[0]))

        preds_arr[idx, :, :] = infer(test_dataset[idx, :, :, :], model,
                                     input_shape,
                                     step_size=args.window_step,
                                     rm_cutoff=args.footprint_threshold)

    geojson_output_dir = os.path.join(args.output_dir, 'output_geojson')
    geotiff_path = os.path.join(args.test_dataset_dir, 'geotiffs')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(geojson_output_dir):
        os.mkdir(geojson_output_dir)
    else:
        warnings.warn('GeoJSON output path already exists. If files with the' +
                      ' same name as outputs already exist there,' +
                      ' this will cause errors.')
    chip_summary_list = []
    for angle_idx in range(test_dataset.shape[0]):
        angle_gj_path = os.path.join(geojson_output_dir,
                                     test_collect_names[angle_idx])
        if not os.path.exists(angle_gj_path):
            os.mkdir(angle_gj_path)
        for chip_idx in range(test_dataset.shape[1]):
            im_fname = [f for f in os.listdir(geotiff_path)
                        if test_fnames[chip_idx] in f][0]
            raw_test_im = rasterio.open(os.path.join(geotiff_path, im_fname))
            preds_test = preds_arr[angle_idx, chip_idx, :, :] > 0.5
            preds_test = preds_test.astype('uint8')
            pred_geojson_path = os.path.join(
                angle_gj_path, test_collect_names[angle_idx] + '_' +
                str(test_fnames[chip_idx]) + '.json'
                )
            try:
                preds_geojson = cLT.createGeoJSONFromRaster(
                    pred_geojson_path, preds_test,
                    raw_test_im.profile['transform'],
                    raw_test_im.profile['crs']
                    )
            except ValueError:
                print('Warning: Empty prediction array for ' +
                      'angle {}, chip {}'.format(str(test_angles[angle_idx]),
                                                 str(test_fnames[chip_idx])))
            chip_summary = {'chipName': im_fname,
                            'geoVectorName': pred_geojson_path,
                            'imageId': test_collect_names[angle_idx] + '_' +
                            test_fnames[chip_idx]}
            chip_summary_list.append(chip_summary)
    csv_output_path = os.path.join(args.output_dir, 'predictions.csv')
    cLT.createCSVSummaryFile(chip_summary_list, csv_output_path,
                             rasterChipDirectory=geotiff_path,
                             createProposalsFile=True,
                             competitionType='buildings',
                             pixPrecision=2)
