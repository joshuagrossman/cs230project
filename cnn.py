import os
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pretty_midi
import constants
import csv
import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def create_model():
    """
    Creates a model with two sets of alternating conv and pooling layers, which outputs to 88
    nodes via a sigmoid activation function. Each of the 88 output values is in the range (0, 1)
    """
    model = Sequential()

    # Dimension: (288, 5, 1)
    model.add(Conv2D(filters=10,
                     kernel_size=(9, 2),
                     strides=(1, 1),
                     padding='SAME',
                     activation='relu',
                     input_shape=(288, 5, 1)))
    # Dimension: (288, 5, 10)
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(2, 1)))
    # Dimension: (143, 4, 10)
    model.add(Conv2D(20, kernel_size=(5, 2), strides=(1, 1), activation='relu'))
    # Dimension: (139, 3, 20)
    model.add(MaxPooling2D(pool_size=(4, 1), strides=(4, 1)))
    # Dimension: (34, 3, 20)
    model.add(Flatten())
    # Dimension: (2100)
    model.add(Dense(1000))
    model.add(Dense(500))
    model.add(Dense(88))
    model.add(Activation('sigmoid'))

    return model

def train_model(x_train, y_train, x_test, y_test):
    """
    Trains CNN and evaluates it on test set. Checkpoints are saved in a directory.

    Inspired by https://github.com/chaumifan/DSL_Final
    """

    print('Creating CNN model\n')
    model = create_model()
    print('Compiling CNN model\n')
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    history = AccuracyHistory()
    checkpoint_path = os.path.join(constants.MODEL_CKPT_DIR, 'ckpt.h5')
    checkpoint = ModelCheckpoint(checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=10)
    
    model.fit(x_train,
              y_train,
              validation_split=0.25,
              epochs=constants.EPOCHS,
              batch_size=10,
              verbose=1,
              callbacks=[history, checkpoint])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Train loss: ', score[0])
    print('Train accuracy: ', score[1])

    print(history.acc)

    plt.plot(range(1,101), history.acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('test_loss.png')

def sigmoid_array(x):                                        
    return np.round(1 / (1 + np.exp(-x)))


def run_cnn(cqt_slice_paths, piano_roll_slice_paths):
    """
    Extracts CNN inputs and then runs training
    """

    print('Reading in CQT slices and piano roll slices\n')
    
    cqt_slices = np.array([np.expand_dims(np.fromfile(cqt_slice_path).reshape((288, 5)), 2) for cqt_slice_path in cqt_slice_paths])

    piano_roll_slices = np.array([np.fromfile(piano_roll_slice_path).reshape((88,)) for piano_roll_slice_path in piano_roll_slice_paths]).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(cqt_slices, piano_roll_slices, test_size=constants.TEST_PERCENTAGE)

    train_model(x_train, y_train, x_test, y_test)
    return
    
class AccuracyHistory(keras.callbacks.Callback):
    """
    Callback for keeping record of training accuracy.

    From http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    """
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def make_fake_data():
    """
    Dummy testing function
    """
    cqt_slice_paths = []
    cqt_slices = []
    piano_roll_slice_paths = []

    for i in range(10):
        filename = 'fake_cqt_slice_%d.csv' % i
        cqt_slice_paths.append(filename)
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            a = np.random.randint(0, 100, (352, 9))
            cqt_slices.append(a)
            for row in a:
                writer.writerow(row)

    w = np.random.random((3168, 88))
    # input data shape is n x height


    for i in range(10):
        filename = 'fake_piano_roll_%d.csv' % i
        piano_roll_slice_paths.append(filename)
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            curr_x = cqt_slices[i]
            curr_x = np.reshape(curr_x, (1, 3168))
            curr_y = np.matmul(curr_x, w).reshape((88, 1))
            curr_y = sigmoid_array(curr_y)

            for row in curr_y:
                writer.writerow(row)

    return cqt_slice_paths, piano_roll_slice_paths

def sigmoid_array(x):                                        
    return np.round(1 / (1 + np.exp(-x)))

def run_cnn(cqt_slice_paths, piano_roll_slice_paths):
    """
    Extracts CNN inputs and then runs training
    """
    print('Reading in CQT slices and piano roll slices\n')
    
    x = np.zeros((num_slices, constants.CONTEXT_WINDOW_ROWS, constants.CQT_SLICE_WIDTH_IN_PIXELS, 1))
    y = np.zeros((num_slices, constants.NUM_KEYS))
    for i in range(num_slices):
        cqt_slice = np.fromfile(cqt_slice_paths[i]).reshape((constants.CONTEXT_WINDOW_ROWS, constants.CQT_SLICE_WIDTH_IN_PIXELS, 1))
        pr_slice = np.fromfile(piano_roll_slice_paths[i]).reshape((constants.NUM_KEYS))
        x[i, :, :, :] = cqt_slice
        y[i, :] = pr_slice

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    train_model(x_train, y_train, x_test, y_test)

# Test model with dummy data
# cqt_slice_paths, piano_roll_slice_paths = make_fake_data()

# cqt_prefix = "Train-CQT-CNN-Slices-In/Prelude_No._1_Slice_"
# pr_prefix = "Train-Pianoroll-Out/Prelude_No._1_Pianoroll_"
# indices = [100361, 101115, 10140, 101615, 102381, 103125, 10512, 10895,
#             1139, 11638, 12, 12021, 12765, 13136]
# cqt_slice_paths = [cqt_prefix + str(i) + '.csv' for i in indices]
# piano_roll_slice_paths = [pr_prefix + str(i) + '.csv' for i in indices]
# run_cnn(cqt_slice_paths, piano_roll_slice_paths)


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
    
