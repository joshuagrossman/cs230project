from constants import *
import cnn
import numpy as np
import os
import math
import threading
import h5py
import re
from keras.models import Sequential, Model


def get_data(piece):
    assert(is_valid_file(piece, "h5"))

    hf = h5py.File(piece, "r")
    slices = np.array(hf.get("slice_indices"))
    cqt = np.array(hf.get("cqt"))
    pianoroll = np.array(hf.get("pianoroll"))
    slice_index = 0

    # Change the cqt and pianoroll data to reflect a bunch of scales with no signal noise
    n_samples = cqt.shape[1]
    n_samples_pianoroll = pianoroll.shape[1]
    if n_samples != n_samples_pianoroll:
        raise Exception("CQT and pianoroll don't have the same number of samples: %d, %d" % (n_samples, n_samples_pianoroll))

    return slices, cqt, pianoroll, slice_index


def load_best_model(dir=MODEL_CKPT_DIR):
    checkpoint_filenames = os.listdir(dir)
    if not checkpoint_filenames:
        raise Exception("The specified weights directory %s is empty. \
                         Try either training from scratch or specifying a different directory." % dir)

    most_recent = os.path.join(dir, max(checkpoint_filenames))

    model = cnn.create_model()
    print("Loading weights from %s..." % most_recent)
    model.load_weights(most_recent)
    print("Done.")
    return model, most_recent


def predict_pianoroll(model, cqt):
    """
    Transcribe the whole CQT by cutting it up into fixed-length sequences.

    We use plenty of overlap so that we can do "OR" voting.
    """
    n_slices = cqt.shape[1] // CONTEXT_WINDOW_COLS
    n_cols = n_slices * CONTEXT_WINDOW_COLS
    cqt = cqt[:, :n_cols].reshape((CONTEXT_WINDOW_ROWS, n_slices, CONTEXT_WINDOW_COLS)).swapaxes(0, 1)
    # New cqt shape is (n_slices, CONTEXT_WINDOW_ROWS, CONTEXT_WINDOW_COLS)

    seq_sampling_freq = SEQUENCE_LENGTH_IN_SLICES // 3
    n_segments = n_slices // seq_sampling_freq
    n_sequences = n_segments - 2

    # Perform predictions
    print("Predicting %d sequences of pianoroll..." % n_sequences)
    input = np.zeros((n_sequences, SEQUENCE_LENGTH_IN_SLICES, CONTEXT_WINDOW_ROWS, CONTEXT_WINDOW_COLS, 1))
    output = np.zeros((n_sequences, SEQUENCE_LENGTH_IN_SLICES, NUM_KEYS))
    for sequence_index in range(n_sequences):
        sequence = cqt[(sequence_index * seq_sampling_freq):(sequence_index * seq_sampling_freq + SEQUENCE_LENGTH_IN_SLICES), :, :]
        input[sequence_index, :, :, :, 0] = sequence
    for sequence_index in range(n_sequences):
        prediction = model.predict(input[sequence_index:(sequence_index + 1), :, :, :, :])
        binarized = np.where(prediction > PREDICTION_CUTOFF, 1, 0)
        output[sequence_index, :, :] = binarized
        print("Finished %d/%d" % (sequence_index + 1, n_sequences), end="\r")
    print("\nDone.")

    # Stitch together the predictions
    print("Stiching together the predictions...")
    pianoroll = np.zeros((n_slices, NUM_KEYS))

    # Start with the first and last segments
    pianoroll[:seq_sampling_freq, :] = output[0, :seq_sampling_freq, :]
    pianoroll[-seq_sampling_freq:, :] = output[-1, -seq_sampling_freq:, :]

    # Next do the second and second to last segments. 
    # These each have only two "votes", so no voting is actually necessary.
    pianoroll[seq_sampling_freq:(2 * seq_sampling_freq), :] = output[0, seq_sampling_freq:(2 * seq_sampling_freq), :]
    pianoroll[-(2 * seq_sampling_freq):-seq_sampling_freq, :] = output[-1, -(2 * seq_sampling_freq):-seq_sampling_freq, :]

    # Do the rest of the segments
    for seq_index in range(1, n_segments - 3):
        prev_seq_segment = output[seq_index - 1, -seq_sampling_freq:, :]
        curr_seq_segment = output[seq_index, seq_sampling_freq:(2 * seq_sampling_freq), :]
        next_seq_segment = output[seq_index + 1, :seq_sampling_freq, :]

        notes = np.logical_or(prev_seq_segment, np.logical_or(curr_seq_segment, next_seq_segment))
        pianoroll[((seq_index + 1)*seq_sampling_freq):((seq_index + 2)*seq_sampling_freq), :] = notes
    print("Done.")

    return pianoroll


def get_notes(pianoroll):
    """
    Convert pianoroll to a list of notes for MIDI generation.
    """
    completed_notes = []
    active_notes = np.zeros((NUM_KEYS,)) # each element will be the slice_index it starts at
    prev_slice = np.zeros((NUM_KEYS,))

    for slice_index in range(1, pianoroll.shape[0] + 1): # offset by 1 so that anything other than 0 is True
        slice = pianoroll[slice_index - 1, :]
        if not np.array_equal(slice, prev_slice):
            # Notes have changed. Update active_notes
            for key in range(NUM_KEYS):
                if slice[key] and not prev_slice[key]:
                    active_notes[key] = slice_index
                elif prev_slice[key] and not slice[key]:
                    note_name = KEYBOARD[key]
                    start = (active_notes[key] - 1) / SLICE_SAMPLING_RATE_IN_HZ # because of the offset
                    end = (slice_index - 1) / SLICE_SAMPLING_RATE_IN_HZ         # because of the offset
                    completed_notes.append((note_name, start, end))
                    active_notes[key] = 0
            prev_slice = slice

    return completed_notes


def is_valid_file(filepath, extension):
    """
    Checks if a file has a valid extension.
    """
    file_re = extension
    
    if extension=="wav":
        file_re = re.compile("\.wav$")
        
    if extension=="midi":
        file_re = re.compile("\.midi$")
        
    if extension=="h5":
        file_re = re.compile("\.h5$")
    
    if not file_re.search(filepath):
        return False
    return True