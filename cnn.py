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
import constants
import csv
import pdb
import argparse
import threading


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


def initialize_generator_vars(piece_index):
    piece = pieces[piece_index]
    hf = h5py.File(piece, "r")
    slices = np.array(hf.get("slice_indices"))
    cqt = np.array(hf.get("cqt"))
    pianoroll = np.array(hf.get("pianoroll"))
    slice_index = 0

    return piece, slices, cqt, pianoroll, slice_index


@threadsafe_generator
def generator(pieces, batch_size=constants.BATCH_SIZE):
    """
    Yield one batch at a time of inputs to the CNN.
    """
    piece_index = 0
    piece, piece_slices, piece_cqt, piece_pianoroll, slice_index = get_piece(piece_index)

    while True:
        # In this iteration of the loop, yield a single batch of sequences
        batch_X = np.zeros((batch_size, constants.SEQUENCE_LENGTH_IN_SLICES, constants.CONTEXT_WINDOW_ROWS, constants.CONTEXT_WINDOW_COLS, 1))
        batch_Y = np.zeros((batch_size, constants.SEQUENCE_LENGTH_IN_SLICES, constants.NUM_KEYS))
        reached_end_of_dataset = False
        for batch_sequence_index in range(batch_size):
            if slice_index + constants.SEQUENCE_LENGTH_IN_SLICES > piece_slices:
                # We can't make another full sequence with this piece
                if piece_index + 1 >= len(pieces):
                    # We've reached the end of an epoch--don't yield these incomplete batches
                    reached_end_of_dataset = True
                    break

                # Skipping to the next piece
                piece_index += 1
                piece, piece_slices, piece_cqt, piece_pianoroll, slice_index = get_piece(piece_index=piece_index)

            # Continue constructing the batch by constructing a sequence and adding it to the batch
            sequence_slices = piece_slices[slice_index:(slice_index + constants.SEQUENCE_LENGTH_IN_SLICES)]
            sequence_cqt = np.zeros((constants.SEQUENCE_LENGTH_IN_SLICES, constants.CONTEXT_WINDOW_ROWS, constants.CONTEXT_WINDOW_COLS))
            sequence_pianoroll = np.zeros((constants.SEQUENCE_LENGTH_IN_SLICES, constants.NUM_KEYS))
            for sequence_index, slice in enumerate(sequence_slices):
                [start, end] = slice
                sequence_cqt[sequence_index, :, :] = piece_cqt[start:end]
                sequence_pianoroll[sequence_index, :] = piece_pianoroll[start:end]

            # Add new sequence to the batches
            batch_X[batch_sequence_index, :, :, :, 0] = sequence_cqt
            batch_Y[batch_sequence_index, :, :] = sequence_pianoroll

            # Increment slice index and go to the next sequence
            slice_index += constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES

        # Batch is done being constructed, yield to CNN
        if reached_end_of_dataset:
            # We weren't able to fill out this batch, just reset everything and let the while-loop restart
            piece, piece_slices, piece_cqt, piece_pianoroll, slice_index = get_piece(piece_index=0)
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
        num_sequences = (num_slices + constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES - constants.SEQUENCE_LENGTH_IN_SLICES) \
            // constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES # trim so that there's a whole number of sequences
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
    model.add(TimeDistributed(conv, input_shape=(constants.SEQUENCE_LENGTH_IN_SLICES, constants.CONTEXT_WINDOW_ROWS, constants.CONTEXT_WINDOW_COLS, 1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2), strides=(2, 1))))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(20, 2), strides=(10, 1), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2), strides=(2, 1))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(500)))
    model.add(LSTM(500, input_shape=(constants.SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True))
    model.add(LSTM(200, input_shape=(constants.SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True))
    model.add(TimeDistributed(Dense(88, activation = "sigmoid")))

    return model


def train_model(pieces):
    """
    Trains CNN and evaluates it on test set. Checkpoints are saved in a directory.

    Inspired by https://github.com/chaumifan/DSL_Final
    """

    print('Creating CNN model\n')
    model = create_model()
    print('Compiling CNN model\n')
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    print(model.summary())

    history = AccuracyHistory()
    checkpoint_path = os.path.join(constants.MODEL_CKPT_DIR, 'ckpt.h5')
    checkpoint = ModelCheckpoint(checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=10)
    
    boundary = (1.0 - constants.VALIDATION_SPLIT) * len(pieces)
    train_pieces = pieces[:boundary]
    valid_pieces = pieces[boundary:]

    num_train_samples = num_samples(train_pieces)
    num_valid_samples = num_samples(valid_pieces)

    model.fit_generator(generator=generator(train_pieces),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=generator(valid_pieces),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=constants.NUM_EPOCHS,
                        batch_size=constants.BATCH_SIZE,
                        verbose=1,
                        callbacks=[history, checkpoint])

    return model, history


def evaluate_model(x_test, y_test, model, history):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Train loss: ', score[0])
    print('Train accuracy: ', score[1])

    print(history.acc)

    plt.plot(range(1, constants.NUM_EPOCHS + 1), history.acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('test_loss.png')



def get_paths_matrix(paths):
    """
    Gets a matrix of paths (strings) in the shape of the CNN feed.

    These paths will later used to retrieve slices
    """

    paths_vector = np.array(paths)
    num_slices = len(paths_vector)
    num_sequences = (num_slices + constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES - constants.SEQUENCE_LENGTH_IN_SLICES) \
        // constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES # trim so that there's a whole number of sequences

    paths_matrix = np.zeros((num_sequences, constants.SEQUENCE_LENGTH_IN_SLICES), dtype=object)
    for i in range(num_sequences):
        paths_matrix[i, :] = paths_vector[(i * constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES):(i * constants.SEQUENCE_SAMPLE_FREQ_IN_SLICES + constants.SEQUENCE_LENGTH_IN_SLICES)]

    return paths_matrix


def cqt_slice_from_path(path):
    """
    Gets the CQT matrix stored in binary at the specified path.
    """

    file_contents = np.fromfile(path)
    cqt_slice = file_contents.reshape((constants.CONTEXT_WINDOW_ROWS, constants.CONTEXT_WINDOW_COLS))
    return cqt_slice


def cqt_slices(paths):
    """
    Returns a tensor pertaining to the full model CQT input.
    """

    # TODO: divide paths into songs
    paths_matrix = get_paths_matrix(paths)
    
    # Convert paths into actual arrays
    slice_tensor = np.zeros((paths_matrix.shape[0], paths_matrix.shape[1], constants.CONTEXT_WINDOW_ROWS, constants.CONTEXT_WINDOW_COLS, 1))
    for i in range(paths_matrix.shape[0]):
        for j in range(paths_matrix.shape[1]):
            cqt_slice = cqt_slice_from_path(paths_matrix[i, j])
            slice_tensor[i, j, :, :, 0] = cqt_slice

    return slice_tensor


def pianoroll_from_path(path):
    """
    Gets the pianoroll stored in binary at the specified path.
    """

    file_contents = np.fromfile(path)
    pianoroll_slice = file_contents.reshape((constants.NUM_KEYS,)).astype(int)
    return pianoroll_slice


def pianoroll_slices(paths):
    """
    Returns a tensor pertaining to the full model pianoroll input.
    """

    # TODO: divide paths into songs
    paths_matrix = get_paths_matrix(paths)
    
    # Convert paths into actual arrays
    slice_tensor = np.zeros((paths_matrix.shape[0], paths_matrix.shape[1], constants.NUM_KEYS))
    for i in range(paths_matrix.shape[0]):
        for j in range(paths_matrix.shape[1]):
            pianoroll_slice = pianoroll_from_path(paths_matrix[i, j])
            slice_tensor[i, j, :] = pianoroll_slice

    return slice_tensor

    
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
    
    model = restore_model(constants.CHECKPOINT_EPOCH, constants.CHECKPOINT_VAL_LOSS)
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

    outPianorollPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, midiFilename).replace("\\", "/")
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
