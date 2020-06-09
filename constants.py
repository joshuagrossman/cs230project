import os

# TODO: fold in golden pianoroll for train (1 pixel wide) and test (many pixels (5ish seconds) wide)

"""
Directories
"""
MODEL_CKPT_DIR = "Models"

# Start with Z so that any "resumes" come lexographically before this one
CHECKPOINT_FILE_TEMPLATE = "Z.{epoch:02d}-{val_loss:.2f}.hdf5"

PREDICTED_HF_NAME = "predicted.h5"

"""
Misc
"""
KEYBOARD = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", \
		    "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", \
		    "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", \
		    "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", \
		    "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", \
		    "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", \
		    "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", \
		    "A7", "A#7", "B7", "C8"]

PREDICTION_CUTOFF = 0.5

"""
Numerical file parameters
"""
BINS_PER_OCTAVE = 36
FILTER_SCALE = 0.5
HOP_LENGTH = 128
CQT_SNAPSHOT_WIDTH_IN_SECONDS = 5

WAV_SAMPLING_RATE_IN_HZ = 44100
CQT_SAMPLING_RATE_IN_HZ = WAV_SAMPLING_RATE_IN_HZ / HOP_LENGTH
SLICE_SAMPLING_RATE_IN_HZ = CQT_SAMPLING_RATE_IN_HZ / CQT_SNAPSHOT_WIDTH_IN_SECONDS

CQT_SLICE_RADIUS_IN_PIXELS = 2
CQT_SLICE_OFFSET_IN_PIXELS = 3
CQT_SLICE_WIDTH_IN_PIXELS = 1 + 2 * CQT_SLICE_RADIUS_IN_PIXELS


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
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SEQUENCE_LENGTH_IN_SLICES = 360 # (~5 seconds), sequence length must be divisible by 3
SEQUENCE_SAMPLE_FREQ_IN_SLICES = 720

TRAIN_VALID_TEST_SPLIT = (0.5, 0.25, 0.25)

"""
File conversion parameters
"""
NOISE_PARAMETER = 0.01