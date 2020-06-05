import os
import numpy as np
import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pretty_midi
from constants import *
import csv
import pdb
import argparse
import threading
import time
import math
import h5py
import librosa, librosa.display


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_data(piece):
    hf = h5py.File(piece, "r")
    slices = np.array(hf.get("slice_indices"))
    cqt = np.array(hf.get("cqt"))
    pianoroll = np.array(hf.get("pianoroll"))
    slice_index = 0

    return slices, cqt, pianoroll, slice_index


@threadsafe_generator
def generator(pieces, batch_size=BATCH_SIZE):
    """
    Yield one batch at a time of inputs to the CNN.
    """
    piece_index = 0
    piece_slices, piece_cqt, piece_pianoroll, slice_index = get_data(pieces[piece_index])

    while True:
        # In this iteration of the loop, yield a single batch of sequences
        batch_X = np.zeros((batch_size, SEQUENCE_LENGTH_IN_SLICES, CONTEXT_WINDOW_ROWS, CONTEXT_WINDOW_COLS, 1))
        batch_Y = np.zeros((batch_size, SEQUENCE_LENGTH_IN_SLICES, NUM_KEYS))
        reached_end_of_dataset = False
        for sequence_index in range(batch_size):
            if slice_index + SEQUENCE_LENGTH_IN_SLICES > piece_slices.shape[0]:
                # We can't make another full sequence with this piece
                if piece_index == len(pieces) - 1:
                    # We've reached the end of an epoch--don't yield these incomplete batches
                    reached_end_of_dataset = True
                    break

                # Skipping to the next piece
                piece_index += 1
                piece_slices, piece_cqt, piece_pianoroll, slice_index = get_data(pieces[piece_index])

            # Construct a sequence and add it to the batch
            sequence_slices = piece_slices[slice_index:(slice_index + SEQUENCE_LENGTH_IN_SLICES)]
            sequence_cqt = np.zeros((SEQUENCE_LENGTH_IN_SLICES, CONTEXT_WINDOW_ROWS, CONTEXT_WINDOW_COLS))
            sequence_pianoroll = np.zeros((SEQUENCE_LENGTH_IN_SLICES, NUM_KEYS))
            for i, slice in enumerate(sequence_slices):
                [start, end] = slice
                sequence_cqt[i, :, :] = piece_cqt[:, start:end]
                sequence_pianoroll[i, :] = np.max(piece_pianoroll[:, start:end], axis=1)

            # Add new sequence to the batches
            batch_X[sequence_index, :, :, :, 0] = sequence_cqt
            batch_Y[sequence_index, :, :] = sequence_pianoroll

            # Increment slice index and go to the next sequence
            slice_index += SEQUENCE_SAMPLE_FREQ_IN_SLICES

        # Batch is done being constructed, yield to CNN
        if reached_end_of_dataset:
            # We weren't able to fill out this batch, just reset everything and let the while-loop restart
            piece_index = 0
            piece_slices, piece_cqt, piece_pianoroll, slice_index = get_data(pieces[piece_index])
        else:
            yield batch_X, batch_Y


def num_samples(dataset):
    """
    Find the number of sequences given by the dataset.
    """
    total_num_sequences = 0
    for piece in dataset:
        hf = h5py.File(piece, "r")
        num_slices = np.array(hf.get("slice_indices")).shape[0]
        num_sequences = (num_slices + SEQUENCE_SAMPLE_FREQ_IN_SLICES - SEQUENCE_LENGTH_IN_SLICES) \
            // SEQUENCE_SAMPLE_FREQ_IN_SLICES # trim so that there's a whole number of sequences
        total_num_sequences += num_sequences

    return total_num_sequences


def create_model():
    """
    Specifies a simple CNN/LSTM recurrent neural network.
    """
    model = Sequential()

    conv = Conv2D(filters=20,
                 kernel_size=(20, 2),
                 strides=(1, 1),
                 padding='SAME',
                 activation='relu')
    model.add(TimeDistributed(conv, input_shape=(SEQUENCE_LENGTH_IN_SLICES, CONTEXT_WINDOW_ROWS, CONTEXT_WINDOW_COLS, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2), strides=(2, 1))))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(20, 2), strides=(10, 1), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2), strides=(2, 1))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(500)))
    model.add(LSTM(500, input_shape=(SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True, recurrent_dropout=DROPOUT_P))
    model.add(LSTM(200, input_shape=(SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True))
    model.add(TimeDistributed(Dense(88, activation = "sigmoid")))

    return model


def train_model(pieces):
    """
    Trains CNN and evaluates it on test set. Checkpoints are saved in a directory.

    Inspired by https://github.com/chaumifan/DSL_Final
    """

    model = create_model()
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    print(model.summary())

    history = AccuracyHistory()
    checkpoint_path = os.path.join(MODEL_CKPT_DIR, 'ckpt.h5')
    checkpoint = ModelCheckpoint(checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=1)
    
    boundary = math.floor((1.0 - VALIDATION_SPLIT) * len(pieces))
    train_pieces = pieces[:boundary]
    valid_pieces = pieces[boundary:]

    num_train_samples = num_samples(train_pieces)
    num_valid_samples = num_samples(valid_pieces)

    print("Number of training batches:", num_train_samples // BATCH_SIZE)
    print("Number of validation batches:", num_valid_samples // BATCH_SIZE)

    model.fit_generator(generator=generator(train_pieces),
                        steps_per_epoch=num_train_samples // BATCH_SIZE,
                        validation_data=generator(valid_pieces),
                        validation_steps=num_valid_samples // BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        callbacks=[history, checkpoint])

    return model, history


def evaluate_model(x_test, y_test, model, history):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Train loss: ', score[0])
    print('Train accuracy: ', score[1])

    print(history.acc)

    plt.plot(range(1, NUM_EPOCHS + 1), history.acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('test_loss.png')

    
class AccuracyHistory(keras.callbacks.Callback):
    """
    Callback for keeping record of training accuracy.

    From http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    """
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def restore_model(epoch, val_loss):
    checkpoint_path = "Models/"    
    model = create_model()
    file = checkpoint_path + 'ckpt.h5weights.%d-%.2f.hdf5' % (epoch, val_loss)
    model.load_weights(file)
    return model


def make_predictions(cqt_data, midiFilename):
    """
    Given a set of cqt_slice_paths, runs predict on each slice and adds CNN
    output as a column to pianoroll prediction. Writes pianoroll prediction to file.
    """
    
    model = restore_model(CHECKPOINT_EPOCH, CHECKPOINT_VAL_LOSS)
    print("CNN model restored.")
    cqt_array = np.array(cqt_data)

    predictions = None
    for i in range(cqt_array.shape[1]):
        small_slice = cqt_array[:,i]
        repeated_slice = np.asarray([[row] * 9 for row in small_slice])
        cqt_slice = np.expand_dims(repeated_slice, 3)
        cqt_slice = np.expand_dims(cqt_slice, 0)
        pred = model.predict(cqt_slice)
        bool_array = pred >= 0.5
        result = bool_array.astype(int)
        if predictions is None:
            predictions = result.T
        else:
            predictions = np.hstack((predictions, result.T))
    print("Pianoroll predictions made.")

    outPianorollPath = os.path.join(TEST_PIANOROLL_OUT_DIR, midiFilename).replace("\\", "/")
    np.savetxt(outPianorollPath, predictions, fmt='%i', delimiter='\t')
    

if __name__ == '__main__':
    """
    If this file is run directly, it tests input/output.

    Usage:

    python cnn.py <cqt-dir> <pianoroll-dir>
    """
    parser = argparse.ArgumentParser(description = 'Test CNN')
    parser.add_argument('cqt_dir', help='Directory of CQT slices')
    parser.add_argument('pianoroll_dir', help='Directory of corresponding pianorolls')
    args = parser.parse_args()

    cqtSlicePaths = [os.path.join(args.cqt_dir, cqt_slice_path).replace("/", "\\") for cqt_slice_path in os.listdir(args.cqt_dir)]
    pianoPaths = [os.path.join(args.pianoroll_dir, pianoroll_path).replace("/", "\\") for pianoroll_path in os.listdir(args.pianoroll_dir)]

    print("TESTING I/O")
    print("===========")
    print("Loading %d CQT inputs..." % len(cqtSlicePaths))
    X = cqt_slices(cqtSlicePaths)
    print("Done loading inputs. Shape is", X.shape)

    assert(len(cqtSlicePaths) == len(pianoPaths))

    print("Loading %d pianoroll inputs..." % len(pianoPaths))
    Y = pianorolls(pianoPaths)
    print("Done loading inputs. Shape is", Y.shape)
