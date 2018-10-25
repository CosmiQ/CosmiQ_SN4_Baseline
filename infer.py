import argparse
from keras.models import load_model
import os
import rasterio
import numpy as np
from spacenetutilities.labeltools import coreLabelTools as cLT
from functools import partial, update_wrapper
from keras import backend as K
from skimage.measure import label
from cw_eval.baseeval import eval_base
import warnings
import pandas as pd


def infer(image_array, model, model_input_shape, step_size, rm_cutoff=0):
    '''Run inference on a large image by tiling.'''
    x_steps = int(np.ceil((image_array.shape[1]-model_input_shape[1])/step_size)) + 1
    y_steps = int(np.ceil((image_array.shape[0]-model_input_shape[0])/step_size)) + 1
    training_arr = np.empty((x_steps*y_steps, model_input_shape[0], model_input_shape[1], image_array.shape[2]))
    raw_inference_arr = np.empty((x_steps*y_steps, image_array.shape[0], image_array.shape[1]))
    raw_inference_arr[:] = np.nan
    counter = 0
    subarr_indices = []
    y_ind = 0
    while y_ind < image_array.shape[0] + step_size - model_input_shape[0] - 1:
        if y_ind + model_input_shape[0] > image_array.shape[0]:
            y_ind = image_array.shape[0] - model_input_shape[0]
        x_ind = 0
        while x_ind < image_array.shape[1] + step_size - model_input_shape[1] - 1:
            if x_ind + model_input_shape[0] > image_array.shape[1]:
                x_ind = image_array.shape[1] - model_input_shape[1]
            training_arr[counter, :, :, :] = image_array[y_ind:y_ind+model_input_shape[0],
                                                         x_ind:x_ind+model_input_shape[1], :]
            subarr_indices.append((y_ind, x_ind))
            x_ind += step_size
            counter += 1
        y_ind += step_size
    predictions = model.predict(training_arr)
    for idx in range(len(subarr_indices)):
        raw_inference_arr[
            idx,
            subarr_indices[idx][0]:subarr_indices[idx][0] + model_input_shape[0],
            subarr_indices[idx][1]:subarr_indices[idx][1] + model_input_shape[1]
            ] = predictions[idx, :, :, 0]
    final_preds = np.nanmean(raw_inference_arr, axis=0) > 0.5
    if rm_cutoff:
        labels = label(final_preds)
        labs, cts = np.unique(labels, return_counts=True)
        labels[np.isin(labels, labs[cts < rm_cutoff])] = 0
        final_preds = labels > 0
    return final_preds


def hybrid_bce_jaccard(y_true, y_pred, weight=0.25):
    """Hybrid loss function combining bce and log(jaccard) loss."""
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = intersection / (sum_ - intersection)
    return (1-weight)*K.binary_crossentropy(y_true, y_pred) + weight*(1-jac)


def precision(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise precision TP/(TP + FP).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2), dtype='float32'))

    # get TP + FN = pred_pos
    pred_pos_ct = K.sum(pred_pos) + K.epsilon()
    precision = true_pos/pred_pos_ct
    return precision


def recall(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise recall TP/(TP + FN).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2), dtype='float32'))
    truth_ct = K.sum(truth) + K.epsilon()
    if truth_ct == 0:
        return 0
    recall = true_pos/truth_ct
    return recall


def w_binary_crossentropy(y_true, y_pred, weight):
    if weight == 1:
        return K.binary_crossentropy(y_pred, y_true)
    weight_mask = K.ones_like(y_true)
    class_two = K.equal(y_true, weight_mask)
    class_two = K.cast(class_two, 'float32')
    if weight < 1:
        class_two = class_two*(1-weight)
        final_mask = weight_mask - class_two
    elif weight > 1:
        class_two = class_two*(weight-1)
        final_mask = weight_mask + class_two
    return K.binary_crossentropy(y_pred, y_true) * final_mask


def f1_score(y_true, y_pred):
    # from basiss: https://github.com/CosmiQ/basiss
    '''https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model'''

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    c1 = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


model_custom_objs = {'hybrid_bce_jaccard': hybrid_bce_jaccard,
                     'precision': precision, 'recall': recall}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='')
    parser.add_argument('--eval_dataset_dir', '-e', type=str, default='eval')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='eval_output')
    parser.add_argument('--verbose', '-v', action='store_const', const=True,
                        default=False)
    parser.add_argument('--angle_set', '-as', type=str, default='')
    parser.add_argument('--angle', '-a', type=int, nargs='+', default=0)
    parser.add_argument('--n_chips', '-n', type=int, default=0)
    parser.add_argument('--randomize_chips', '-r', action='store_const',
                        const=True, default=False)
    parser.add_argument('--size_threshold', '-st', type=int, default=0)
    parser.add_argument('--make_csv', '-c', action='store_const', const=True,
                        default=False)
    args = parser.parse_args()


    # read in model
    model = load_model(args.model_path,
                       custom_objects=model_custom_objs)

    # read in eval dataset
    eval_arr = np.load(os.path.join(args.eval_dataset_dir, 'eval_arr.npy'),
                       mmap_mode='r')
    eval_angles = np.load(os.path.join(args.eval_dataset_dir,
                                       'eval_angles.npy'))
    eval_angle_names = np.load(os.path.join(args.eval_dataset_dir,
                                            'eval_angle_names.npy'))
    eval_chips = np.load(os.path.join(args.eval_dataset_dir,
                                      'eval_chips.npy'))

    # subset eval angles
    eval_angle_mask = np.array([True for i in range(eval_arr.shape[0])])
    if args.angle_set:
        if args.angle_set == 'nadir':
            eval_angle_mask = eval_angles < 26
        elif args.angle_set == 'offnadir':
            eval_angle_mask = np.logical_and(eval_angles > 25,
                                             eval_angles < 40)
        elif args.angle_set == 'faroffnadir':
            eval_angle_mask = eval_angles > 40
    elif args.angle:
        eval_angle_mask = np.isin(eval_angles, args.angle)
    if not np.all(eval_angle_mask):
        eval_arr = eval_arr[eval_angle_mask]
        eval_angles = eval_angles[eval_angle_mask]

    # subset eval chips
    if args.randomize_chips:
        # get shuffled ax order
        ax_shuffle = np.random.shuffle(np.arange(eval_arr.shape[1]))
        eval_arr = eval_arr[:, ax_shuffle, :, :, :]
        eval_chips = eval_chips[ax_shuffle]
    if args.n_chips:
        eval_arr = eval_arr[:, :args.n_chips, :, :, :]
        eval_chips = eval_chips[:args.n_chips]

    # perform inference for each sub-image
    preds_arr = np.empty(eval_arr.shape[0:-1])
    for angle in range(eval_arr.shape[0]):
        if args.verbose:
            print('Processing angle #{}'.format(angle))
        input_shape = model.layers[0].input_shape[1:]
        for idx in range(eval_arr.shape[1]):
            preds_arr[angle, idx, :, :] = infer(
                eval_arr[angle, idx, :, :, :], model, input_shape, 64,
                rm_cutoff=args.size_threshold)
            if args.verbose:
                print('    Image #{} inference completed'.format(idx))

    geojson_output_dir = os.path.join(args.output_dir, 'output_geojson')
    geotiff_path = os.path.join(args.eval_dataset_dir, 'geotiffs')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(geojson_output_dir):
        os.mkdir(geojson_output_dir)
    else:
        warnings.warn('GeoJSON output path already exists. If files with the' +
                      ' same name as outputs already exist there,' +
                      ' this will cause errors.')
    chip_summary_list = []
    for angle_idx in range(eval_arr.shape[0]):
        angle_gj_path = os.path.join(geojson_output_dir,
                                     str(eval_angle_names[angle_idx]))
        if not os.path.exists(angle_gj_path):
            os.mkdir(angle_gj_path)
        for chip_idx in range(eval_arr.shape[1]):
            im_fname = [f for f in os.listdir(geotiff_path)
                        if eval_chips[chip_idx] in f][0]
            raw_test_im = rasterio.open(os.path.join(geotiff_path, im_fname))
            preds_test = preds_arr[angle_idx, chip_idx, :, :] > 0.5
            preds_test = preds_test.astype('uint8')
            pred_geojson_path = os.path.join(
                angle_gj_path, str(eval_angle_names[angle_idx]) + '_' +
                str(eval_chips[chip_idx]) + '.json'
                )
            try:
                preds_geojson = cLT.createGeoJSONFromRaster(
                    pred_geojson_path, preds_test,
                    raw_test_im.profile['transform'],
                    raw_test_im.profile['crs']
                    )
            except ValueError:
                print('Warning: Empty prediction array for angle {}, chip {}'.format(
                        str(eval_angles[angle_idx]),
                        str(eval_chips[chip_idx])))
            chip_summary = {'chipName': im_fname,
                            'geoVectorName': pred_geojson_path,
                            'imageId': eval_angle_names[angle_idx] + '_' + eval_chips[chip_idx]}
            chip_summary_list.append(chip_summary)
    csv_output_path = os.path.join(args.output_dir, 'predictions.csv')
    cLT.createCSVSummaryFile(chip_summary_list, csv_output_path,
                             rasterChipDirectory=geotiff_path,
                             createProposalsFile=True,
                             competitionType='buildings',
                             pixPrecision=2)
