import numpy as np
from skimage.measure import label


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
