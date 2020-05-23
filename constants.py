import os
import re

# TODO: fold in golden pianoroll for train (1 pixel wide) and test (many pixels (5ish seconds) wide)

"""
Directories
"""
HOME = os.path.expanduser("~").replace("\\", "/")
ROOT_DIR = HOME + "/Documents/221-machine-music-transcription"

# ROOT_DIR = HOME + "/Documents/Coterm I/CS221/221-machine-music-transcription"

# ROOT_DIR = HOME + "/221-machine-music-transcription"

WTC1_WAV_DIR = ROOT_DIR + "/WTC1-WAV"
WTC2_WAV_DIR = ROOT_DIR + "/WTC2-WAV"

WTC1_MIDI_DIR = ROOT_DIR + "/WTC1-MIDI"
WTC2_MIDI_DIR = ROOT_DIR + "/WTC2-MIDI"

# CQT heatmap slices for CNN training
TRAIN_CQT_CNN_SLICES_IN_DIR = ROOT_DIR + "/Train-CQT-CNN-Slices-In"

# Time series text data for full-WAV CQT
TRAIN_TIME_SERIES_IN_DIR = ROOT_DIR + "/Train-Time-Series-In"

# CNN output on training data, if needed
TRAIN_PIANOROLL_OUT_DIR = ROOT_DIR + "/Train-Pianoroll-Out"

# CNN output on test data
TEST_PIANOROLL_OUT_DIR = ROOT_DIR + "/Test-Pianoroll-Out"

# What the CNN output is supposed to be
TEST_PIANOROLL_GOLDEN_DIR = ROOT_DIR + "/Test-Pianoroll-Golden"

# CNN output, converted to MIDI for human listening
TEST_MIDI_OUT_DIR = ROOT_DIR + "/Test-MIDI-Out"

# CQT heatmap slices for CNN
TEST_CQT_CNN_SLICES_IN_DIR = ROOT_DIR + "/Test-CQT-CNN-Slices-In"

# Time series text data for full-WAV CQT
TEST_TIME_SERIES_IN_DIR = ROOT_DIR + "/Test-Time-Series-In"

MODEL_CKPT_DIR = ROOT_DIR + "/Models"

"""
File name templates
"""
FUGUE_WAV_NAME_TEMPLATE = "(Fugue_No\._%s)[_\.].+"
PRELUDE_WAV_NAME_TEMPLATE = "(Prelude_No\._%s)[_\.].+"

# E.g., Fugue_No._1_Slice_458.png (which means 458 milliseconds in)
CQT_SLICE_NAME_TEMPLATE = "%s_Slice_(%s).csv"

# E.g., Fugue_No._1_Pianoroll_6.png
PIANOROLL_NAME_TEMPLATE = "%s_Pianoroll_(%s).csv"

# E.g., Fugue_No._1.mid
MIDI_OUT_NAME_TEMPLATE = "%s.mid"

RESULTS_PATH = ROOT_DIR + "/numericalResults%s.txt"

"""
Numerical file parameters
"""
CQT_SNAPSHOT_WIDTH_IN_SECONDS = 5
CQT_SNAPSHOT_WIDTH_IN_PIXELS = 1000 # TODO: is it too high? We should have about one solid colored block per pixel
CQT_SNAPSHOT_HEIGHT_IN_PIXELS = 1000


CQT_SAMPLING_RATE = 86.1

CQT_SLICE_WIDTH_IN_PIXELS = 1
CQT_SLICE_HEIGHT_IN_PIXELS = CQT_SNAPSHOT_HEIGHT_IN_PIXELS

TRAIN_PIANOROLL_WIDTH_IN_PIXELS = 1
TEST_PIANOROLL_WIDTH_IN_SECONDS = 5
TEST_PIANOROLL_WIDTH_IN_PIXELS = float(TEST_PIANOROLL_WIDTH_IN_SECONDS) * TRAIN_PIANOROLL_WIDTH_IN_PIXELS / CQT_SLICE_WIDTH_IN_PIXELS * CQT_SNAPSHOT_WIDTH_IN_PIXELS / CQT_SNAPSHOT_WIDTH_IN_SECONDS

"""
Hyperparameters
"""
CONTEXT_WINDOW_ROWS = 352
CONTEXT_WINDOW_COLS = 9
NUM_RGB_CHANNELS = 1

NUM_KEYS = 88
DROPOUT_P = 0.25
BATCH_SIZE = 10
EPOCHS = 100

NUM_PIECES_TO_TEST = 4
TEST_PERCENTAGE = 0.2

CHECKPOINT_EPOCH = 80
CHECKPOINT_VAL_LOSS = 0.49

"""
File conversion parameters
"""
NOISE_PARAMETER = 0.01
DEFAULT_SOUND_FONT = '~/.fluidsynth/default_sound_font.sf2'
SAMPLE_RATE_IN_HZ = 1000
