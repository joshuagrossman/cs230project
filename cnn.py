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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


def train_model(x_train, y_train, x_test, y_test):
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
    
    model.fit(x_train,
              y_train,
              validation_split=0.25,
              epochs=constants.NUM_EPOCHS,
              batch_size=constants.BATCH_SIZE,
              verbose=1,
              callbacks=[history, checkpoint])

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


def run_cnn(cqt_slice_paths, piano_roll_slice_paths):
    """
    Extracts CNN inputs and then runs training.
    """
    assert(len(cqt_slice_paths) == len(piano_roll_slice_paths))

    print("Loading %d CQT inputs..." % len(cqt_slice_paths))
    X = cqt_slices(cqt_slice_paths)
    print("Done loading inputs. Shape is", X.shape)
    print("Loading %d pianoroll inputs..." % len(piano_roll_slice_paths))
    Y = pianoroll_slices(piano_roll_slice_paths)
    print("Done loading inputs. Shape is", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=constants.TEST_PERCENTAGE)

    train_model(X_train, Y_train, X_test, Y_test)

    
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
