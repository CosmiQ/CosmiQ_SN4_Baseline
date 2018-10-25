import os
import argparse
from functools import partial, update_wrapper
import warnings

parser = argparse.ArgumentParser(description='Train the baseline model.')
parser.add_argument('--seed', '-s', type=int, default=1337,
                    help='Randomization seed for initialization and datagen.')
parser.add_argument('--dataset', '-d', type=str, default='all',
                    help='Data to train the model on. Options are ' +
                    'all, nadir, offnadir, or faroffnadir.')
parser.add_argument('--model', '-m', type=str, default='ternausnet',
                    help='Not implemented yet. Model to train the data on. ' +
                    'Only current option is ternausnet.')
parser.add_argument('--output-path', '-o', type=str, default='model.hdf5',
                    help='Path for saving trained model. ' +
                    'Defaults to model.hdf5 in the working directory.')
parser.add_argument('--data-path', '-dp', type=str, default='',
                    help='Path to the dataset .npy files. ' +
                    'Defaults to the current working directory.')
parser.add_argument('--tensorboard-dir', '-t', type=str, default='',
                    help='Path to save logs for TensorBoard. ' +
                    'If not provided, no TensorBoard logs are saved.')

args = parser.parse_args()

# set random seed for numpy and tensorflow
import numpy as np
np.random.seed(args.seed)
import tensorflow as tf
tf.set_random_seed(args.seed)

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from cosmiq_sn4_baseline import DataGenerator, weighted_bce  # TODO: COMPLETE IMPORTS
from cosmiq_sn4_baseline import TerminateOnMetricNaN, ternausnet
from cosmiq_sn4_baseline import foreground_f1_score


def main(dataset, model='ternausnet', data_path='',
         output_path='model.hdf5', tb_dir=''):

    # create a few variables needed later.
    output_dir, model_name = os.path.split(output_path)
    tmp_model_path = os.path.join(output_dir, 'tmp_model.h5')
    tmp_weights_path = os.path.join(output_dir, 'tmp_weights.h5')

    # make sure everything is clean to start:
    if os.path.exists(tmp_model_path):
        warnings.warn('Temp file {} existed before starting. Deleted.'.format(
            tmp_model_path))
        os.remove(tmp_model_path)
    if os.path.exists(tmp_weights_path):
        warnings.warn('Temp file {} existed before starting. Deleted.'.format(
            tmp_weights_path))
        os.remove(tmp_weights_path)

    # specify the path to the training and validation data files
    train_im_path = os.path.join(data_path, dataset + '_training_ims.npy')
    val_im_path = os.path.join(data_path, dataset + '_validation_ims.npy')
    train_mask_path = os.path.join(data_path, 'training_masks.npy')
    val_mask_path = os.path.join(data_path, 'validation_masks.npy')

    batch_size = 32
    n_tries = 50
    early_stopping_patience = 50
    score_cutoff = 0.5
    model_args = {
        'optimizer': 'Adam',
        'input_size': (256, 256, 3),
        'base_depth': 64,
        'lr': 0.0001,
        'loss_weight': 6
    }

    # load in data. don't read entirely into memory - too big.
    train_im_arr = np.load(train_im_path, mmap_mode='r')
    val_im_arr = np.load(val_im_path, mmap_mode='r')
    train_mask_arr = np.load(train_mask_path, mmap_mode='r')
    val_mask_arr = np.load(val_mask_path, mmap_mode='r')

    # create generators for training and validation
    training_gen = DataGenerator(train_im_arr, train_mask_arr,
                                 batch_size=batch_size,
                                 crop=True,
                                 output_x=model_args['input_size'][1],
                                 output_y=model_args['input_size'][0],
                                 flip_x=True, flip_y=True, rotate=True,
                                 rescale_brightness=[0.75, 1.5],
                                 zoom_range=0.2)
    validation_gen = DataGenerator(val_im_arr, val_mask_arr,
                                   batch_size=batch_size,
                                   crop=True,
                                   output_x=model_args['input_size'][1],
                                   output_y=model_args['input_size'][0])
    tries = 0
    have_weights = False
    monitor = 'foreground_f1_score'
    trained = False
    transition_score = 0.8
    # because the all data and faroffnadir data models don't achieve as good
    # f1 score after training with weighted bce, the transition score used
    # (see comment after callbax) must be lower.
    if dataset in ['all', 'faroffnadir']:
        transition_score = 0.65

    while tries < n_tries and trained is False:
        print()
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("TRAINING ATTEMPT: {} OF {}".format(tries+1, n_tries))
        print("LEARNING RATE: {}".format(model_args['lr']))
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print()

        callbax = []
        # the model is first trained with a weighted bce loss function, then
        # once it reaches a good enough f1 score cutoff, it's trained with
        # normal bce to improve accuracy. The metrics monitored reflect this.
        callbax.append(ModelCheckpoint(tmp_model_path, monitor=monitor,
                                       save_best_only=True))
        callbax.append(TerminateOnMetricNaN('val_foreground_f1_score'))
        callbax.append(EarlyStopping(monitor=monitor,
                                     patience=early_stopping_patience,
                                     mode='max'))
        if tb_dir:  # if saving tensorboard logs
            callbax.append(TensorBoard(
                log_dir=os.path.join(tb_dir, model_name)))
        model = ternausnet(**model_args)
        if have_weights:
            model.load_weights(tmp_weights_path)
        hist = model.fit_generator(
            training_gen, validation_data=validation_gen,
            validation_steps=np.floor(val_im_arr.shape[1]/batch_size),
            steps_per_epoch=np.floor(train_im_arr.shape[1]/batch_size),
            epochs=1000, callbacks=callbax)
        print()
        print('validation F1 scores:')
        print(hist.history['val_foreground_f1_score'])
        print()
        best_score = np.nanmax(hist.history[monitor])
        print('best validation {} score: {}'.format(monitor, best_score))
        print('score cutoff: {}'.format(score_cutoff))
        if best_score > score_cutoff:
            print()
            print("!!! ACHIEVED NEXT SCORE THRESHOLD !!!")
            # define loss function for loading model
            lf = partial(weighted_bce,
                         weight=model_args['loss_weight'])
            lf = update_wrapper(lf, weighted_bce)
            best_m = keras.models.load_model(tmp_model_path, custom_objects={
                'weighted_bce': lf,
                'foreground_f1_score': foreground_f1_score})
            best_m.save_weights(tmp_weights_path)
            have_weights = True
            # if training with weighted_bce (monitoring f1_score)
            # and achieve a training set f1_score above threshold,
            # transition to an unweighted bce and monitor accuracy instead.
            if monitor == 'foreground_f1_score' and best_score > transition_score:
                monitor = 'acc'
                best_score = 0.6
                model_args['lr'] = 0.00001
                model_args['loss_weight'] = 1
            elif monitor == 'acc' and best_score > 0.95:
                trained = True
            else:
                score_cutoff = best_score + (1-best_score)*0.25
                model_args['lr'] *= 0.2
            print("Reducing learning rate to {}".format(model_args['lr']))
            print("Increasing score cutoff to {}".format(score_cutoff))
        tries += 1
    model.save(output_path)


if __name__ == '__main__':
    main()
