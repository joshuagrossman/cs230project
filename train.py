from util import *
from constants import *
import cnn
import os
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pretty_midi
import csv
import pdb
import argparse
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


# def n_samples(dataset):
#     """
#     Find the number of sequences given by the dataset.
#     """
#     total_num_sequences = 0
#     for piece in dataset:
#         hf = h5py.File(piece, "r")
#         num_slices = np.array(hf.get("slice_indices")).shape[0]
#         num_sequences = (num_slices + SEQUENCE_SAMPLE_FREQ_IN_SLICES - SEQUENCE_LENGTH_IN_SLICES) \
#             // SEQUENCE_SAMPLE_FREQ_IN_SLICES # trim so that there's a whole number of sequences
#         total_num_sequences += num_sequences

#     return total_num_sequences

def n_samples(dataset, batch_size=BATCH_SIZE):
    result = 0

    piece_index = 0
    piece_slices, piece_cqt, piece_pianoroll, slice_index = get_data(dataset[piece_index])

    for sequence_index in range(batch_size):
        if slice_index + SEQUENCE_LENGTH_IN_SLICES > piece_slices.shape[0]:
            # We can't make another full sequence with this piece
            if piece_index == len(dataset) - 1:
                # We've reached the end of an epoch--don't yield these incomplete batches
                return result

            # Skipping to the next piece
            piece_index += 1
            piece_slices, piece_cqt, piece_pianoroll, slice_index = get_data(dataset[piece_index])

        # Increment slice index and go to the next sequence
        slice_index += SEQUENCE_SAMPLE_FREQ_IN_SLICES
        result += 1

    return result


def train_model(train_pieces, valid_pieces, batch_size=BATCH_SIZE, n_epochs=NUM_EPOCHS):
    """
    Trains CNN and evaluates it on test set. Checkpoints are saved in a directory.

    Inspired by https://github.com/chaumifan/DSL_Final
    """

    model = cnn.create_model()
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    print(model.summary())

    history = AccuracyHistory()
    checkpoint = ModelCheckpoint(os.path.join(MODEL_CKPT_DIR, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 mode='auto',
                                 period=1)

    n_train_samples = n_samples(train_pieces, batch_size=batch_size)
    n_valid_samples = n_samples(valid_pieces, batch_size=batch_size)

    print("Number of training batches:", n_train_samples // batch_size)
    print("Number of validation batches:", n_valid_samples // batch_size)

    model.fit_generator(generator=generator(train_pieces, batch_size=batch_size),
                        steps_per_epoch=n_train_samples // batch_size,
                        validation_data=generator(valid_pieces, batch_size=batch_size),
                        validation_steps=n_valid_samples // batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=[history, checkpoint])

    
class AccuracyHistory(keras.callbacks.Callback):
    """
    Callback for keeping record of training accuracy.

    From http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    """
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
    