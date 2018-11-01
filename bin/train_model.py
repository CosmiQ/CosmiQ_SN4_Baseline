import os
import argparse
import warnings
import numpy as np

parser = argparse.ArgumentParser(description='Train the baseline model.')
parser.add_argument('--data_path', '-d', type=str, default='',
                    help='Path to the directory containing the `train` and ' +
                    '`val` data folders. Defaults to the current working ' +
                    'directory.')
parser.add_argument('--output_path', '-o', type=str, default='model.hdf5',
                    help='Path for saving trained model. ' +
                    'Defaults to model.hdf5 in the working directory.')
parser.add_argument('--subset', '-s', type=str, default='all',
                    help='Data to train the model on. Options are ' +
                    '`all`, `nadir`, `offnadir`, or `faroffnadir`.')
parser.add_argument('--seed', '-e', type=int, default=1337,
                    help='Randomization seed for initialization and datagen.')
parser.add_argument('--model', '-m', type=str, default='ternausnetv1',
                    help='Model architecture. Either `ternausnetv1` or `unet`.' +
                    ' See cosmiq_sn4_baseline.model for architecture details.')
parser.add_argument('--tensorboard_dir', '-t', type=str, default='',
                    help='Path to save logs for TensorBoard. ' +
                    'If not provided, no TensorBoard logs are saved.')

args = parser.parse_args()

# set random seed for numpy and tensorflow
np.random.seed(args.seed)
import tensorflow as tf
tf.set_random_seed(args.seed)
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from cosmiq_sn4_baseline.DataGenerator import FlatDataGenerator
from cosmiq_sn4_baseline.callbacks import TerminateOnMetricNaN
from cosmiq_sn4_baseline.losses import hybrid_bce_jaccard
from cosmiq_sn4_baseline.metrics import precision, recall
from cosmiq_sn4_baseline.models import compile_model

def main(dataset, model='ternausnetv1', data_path='',
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
    train_im_path = os.path.join(data_path, 'train',
                                 dataset + '_train_ims.npy')
    val_im_path = os.path.join(data_path, 'validate',
                               dataset + '_val_ims.npy')
    train_mask_path = os.path.join(data_path, 'train',
                                   dataset + '_train_masks.npy')
    val_mask_path = os.path.join(data_path, 'validate',
                                 dataset + '_val_masks.npy')

    batch_size = 4
    early_stopping_patience = 50
    model_args = {
        'optimizer': 'Nadam',
        'input_size': (512, 512, 3),
        'base_depth': 64,
        'lr': 0.0001
    }
    # reduce base_depth to 32 if using vanilla unet
    if model == 'unet':
        model_args['base_depth'] = 32

    # load in data. don't read entirely into memory - too big.
    train_im_arr = np.load(train_im_path, mmap_mode='r')
    val_im_arr = np.load(val_im_path, mmap_mode='r')
    train_mask_arr = np.load(train_mask_path, mmap_mode='r')
    val_mask_arr = np.load(val_mask_path, mmap_mode='r')

    # create generators for training and validation
    training_gen = FlatDataGenerator(
        train_im_arr, train_mask_arr, batch_size=batch_size, crop=True,
        output_x=model_args['input_size'][1],
        output_y=model_args['input_size'][0],
        flip_x=True, flip_y=True, rotate=True
        )
    validation_gen = FlatDataGenerator(
        val_im_arr, val_mask_arr, batch_size=batch_size, crop=True,
        output_x=model_args['input_size'][1],
        output_y=model_args['input_size'][0]
        )
    monitor = 'val_loss'
    print()
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("                 BEGINNING MODEL TRAINING")
    print("                 MODEL ARCHITECTURE: {}".format(model))
    print("                   OPTIMIZER: {}".format(model_args['optimizer']))
    print("                     DATASET: {}".format(dataset))
    print("                 INPUT SHAPE: {}".format(model_args['input_shape']))
    print("                      BATCH SIZE: {}".format(batch_size))
    print("                   LEARNING RATE: {}".format(model_args['lr']))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print()

    callbax = []
    callbax.append(ModelCheckpoint(tmp_model_path, monitor=monitor,
                                   save_best_only=True))
    callbax.append(TerminateOnMetricNaN('precision'))
    callbax.append(EarlyStopping(monitor=monitor,
                                 patience=early_stopping_patience,
                                 mode='max'))
    if tb_dir:  # if saving tensorboard logs
        callbax.append(TensorBoard(
            log_dir=os.path.join(tb_dir, model_name)))
    lf = hybrid_bce_jaccard
    am = [precision,
          recall]
    model = compile_model(arch=model, loss_func=lf,
                          additional_metrics=am,
                          verbose=True, **model_args)
    model.fit_generator(
        training_gen, validation_data=validation_gen,
        validation_steps=np.floor(val_im_arr.shape[1]/batch_size),
        steps_per_epoch=np.floor(train_im_arr.shape[1]/batch_size),
        epochs=1000, callbacks=callbax
        )
    model.save(output_path)
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("                   MODEL TRAINING COMPLETE!                 ")
    print("   Model located at {}".format(output_path))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")


if __name__ == '__main__':
    main(args.subset, model=args.model, data_path=args.data_path,
         output_path=args.output_path, tb_dir=args.tensorboard_dir)
