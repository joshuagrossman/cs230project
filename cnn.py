from constants import *
import keras
from keras.layers import *
from keras.models import Sequential, Model


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
    model.add(Bidirectional(LSTM(500, input_shape=(SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True, recurrent_dropout=DROPOUT_P)))
    model.add(Bidirectional(LSTM(200, input_shape=(SEQUENCE_LENGTH_IN_SLICES, 500), return_sequences=True)))
    model.add(TimeDistributed(Dense(88, activation = "sigmoid")))

    return model