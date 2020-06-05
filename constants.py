import os
import re

# TODO: fold in golden pianoroll for train (1 pixel wide) and test (many pixels (5ish seconds) wide)

"""
Directories
"""
HOME = os.path.expanduser("~").replace("\\", "/")

#WTC1_WAV_DIR = "/WTC1-WAV"
#WTC2_WAV_DIR = "/WTC2-WAV"

WTC1_MIDI_DIR = "/WTC1-MIDI"
WTC2_MIDI_DIR = "/WTC2-MIDI"

# CQT heatmap slices for CNN training
TRAIN_CQT_CNN_SLICES_IN_DIR = "Train-CQT-CNN-Slices-In"

# Time series text data for full-WAV CQT
TRAIN_TIME_SERIES_IN_DIR = "Train-Time-Series-In"

# CNN output on training data, if needed
TRAIN_PIANOROLL_OUT_DIR = "Train-Pianoroll-Out"

# CNN output on test data
TEST_PIANOROLL_OUT_DIR = "Test-Pianoroll-Out"

# What the CNN output is supposed to be
TEST_PIANOROLL_GOLDEN_DIR = "Test-Pianoroll-Golden"

# CNN output, converted to MIDI for human listening
TEST_MIDI_OUT_DIR = "Test-MIDI-Out"

# CQT heatmap slices for CNN
TEST_CQT_CNN_SLICES_IN_DIR = "Test-CQT-CNN-Slices-In"

# Time series text data for full-WAV CQT
TEST_TIME_SERIES_IN_DIR = "Test-Time-Series-In"

MODEL_CKPT_DIR = "Models"

# corresponds to a max song length of 3 hours
MAX_START_MS_DIGITS = 7

# """
# File name templates
# """
# FUGUE_WAV_NAME_TEMPLATE = "(Fugue_No\._%s)[_\.].+"
# PRELUDE_WAV_NAME_TEMPLATE = "(Prelude_No\._%s)[_\.].+"

# # E.g., Fugue_No._1_Slice_458.png (which means 458 milliseconds in)
# CQT_SLICE_NAME_TEMPLATE = "%s_Slice_(%s).csv"

# # E.g., Fugue_No._1_Pianoroll_6.png
# PIANOROLL_NAME_TEMPLATE = "%s_Pianoroll_(%s).csv"

# # E.g., Fugue_No._1.mid
# MIDI_OUT_NAME_TEMPLATE = "%s.mid"

# RESULTS_PATH = "/numericalResults%s.txt"

"""
Numerical file parameters
"""
CQT_SNAPSHOT_WIDTH_IN_SECONDS = 5
CQT_SNAPSHOT_WIDTH_IN_PIXELS = 1000 # TODO: is it too high? We should have about one solid colored block per pixel
CQT_SNAPSHOT_HEIGHT_IN_PIXELS = 1000

CQT_SAMPLING_RATE = 344.53125 # (which is 44100 / 128)

CQT_SLICE_RADIUS_IN_PIXELS = 2
CQT_SLICE_OFFSET_IN_PIXELS = 3
CQT_SLICE_WIDTH_IN_PIXELS = 1 + 2 * CQT_SLICE_RADIUS_IN_PIXELS

CQT_SLICE_HEIGHT_IN_PIXELS = CQT_SNAPSHOT_HEIGHT_IN_PIXELS

# TRAIN_PIANOROLL_WIDTH_IN_PIXELS = 1
# TEST_PIANOROLL_WIDTH_IN_SECONDS = 5
# TEST_PIANOROLL_WIDTH_IN_PIXELS = float(TEST_PIANOROLL_WIDTH_IN_SECONDS) * TRAIN_PIANOROLL_WIDTH_IN_PIXELS / CQT_SLICE_WIDTH_IN_PIXELS * CQT_SNAPSHOT_WIDTH_IN_PIXELS / CQT_SNAPSHOT_WIDTH_IN_SECONDS

"""
Hyperparameters
"""
#CONTEXT_WINDOW_ROWS = 352
CONTEXT_WINDOW_ROWS = 288
CONTEXT_WINDOW_COLS = 5
NUM_RGB_CHANNELS = 1

NUM_KEYS = 88
DROPOUT_P = 0.25
NUM_EPOCHS = 10
BATCH_SIZE = 128
SEQUENCE_LENGTH_IN_SLICES = 350 # (~5 seconds)
SEQUENCE_SAMPLE_FREQ_IN_SLICES = 175

NUM_PIECES_TO_TEST = 4
TEST_SPLIT = 0.25
VALIDATION_SPLIT = 0.333

# CHECKPOINT_EPOCH = 80
# CHECKPOINT_VAL_LOSS = 0.49

CHECKPOINT_EPOCH = 100
CHECKPOINT_VAL_LOSS = 0.17

BINS_PER_OCTAVE = 36
FILTER_SCALE = 0.5
HOP_LENGTH = 128

"""
File conversion parameters
"""
NOISE_PARAMETER = 0.01
DEFAULT_SOUND_FONT = '~/.fluidsynth/default_sound_font.sf2'
SAMPLE_RATE_IN_HZ = 1000
