from keras import backend as K

def precision(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise precision TP/(TP + FP).

    """
    # count true positives
    true_pos = K.sum(K.cast(K.all(K.stack([K.round(K.clip(y_true, 0, 1)),
                                           K.round(K.clip(y_pred, 0, 1))],
                                          axis=2),
                                  axis=2), dtype='float32'))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_pos/pred_pos
    return precision


def recall(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise recall TP/(TP + FN).

    """
    # count true positives
    true_pos = K.sum(K.cast(K.all(K.stack([K.round(K.clip(y_true, 0, 1)),
                                           K.round(K.clip(y_pred, 0, 1))],
                                          axis=2),
                                  axis=2), dtype='float32'))
    ground_truth = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_pos/ground_truth
    return recall

def f1_score(y_true, y_pred):
    """F1 score for foreground pixels ONLY.

    Calculates pixelwise F1 score for the foreground pixels (mask value == 1).
    Returns NaN if the model does not identify any foreground pixels in the
    image.

    """
    # from basiss: https://github.com/CosmiQ/basiss and
    '''https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model'''

    # Count positive samples.
    c1 = K.sum(K.cast(K.all(K.stack([K.round(K.clip(y_true, 0, 1)),
                                      K.round(K.clip(y_pred, 0, 1))], axis=0)),
                      dtype='float32'))
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
