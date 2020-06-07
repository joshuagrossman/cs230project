from constants import *
from util import *
import evaluate
import cnn
import os
import re
import time
import pdb
import argparse
import h5py
import random


def get_train_valid_test_split(data_dir):
    # Get h5 files corresponding to pieces
    piece_paths = [os.path.join(data_dir, h5_file) \
        for h5_file in os.listdir(data_dir) if "Chamber" not in h5_file]

    # Shuffle it the same every time
    random.Random(7).shuffle(piece_paths)

    # Split pieces into train/test
    train_valid_boundary = round(TRAIN_VALID_TEST_SPLIT[0] * len(piece_paths))
    valid_test_boundary = train_valid_boundary + round(TRAIN_VALID_TEST_SPLIT[1] * len(piece_paths))
    train_pieces = piece_paths[:train_valid_boundary]
    valid_pieces = piece_paths[train_valid_boundary:valid_test_boundary]
    test_pieces = piece_paths[valid_test_boundary:]

    return train_pieces, valid_pieces, test_pieces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Trains a CNN.')
    parser.add_argument('data_dir', help='Directory of h5 files')
    parser.add_argument('action', help='What action you want to perform ("train", "test", or "convert")')
    parser.add_argument('--quick-train', dest='quick_train', action='store_const',
                        help='Use only two pieces for training and validation')
    parser.add_argument('--onsets-only', dest='onsets_only', action='store_const',
                        help='Whether to use onsets_only for testing')
    args = parser.parse_args()

    # Get train/dev/test split
    train_pieces, valid_pieces, test_pieces = get_train_valid_test_split(args.data_dir)

    # Command line argument logic
    if args.action == "train":
        if args.quick_train:
            cnn.train_model(train_pieces[:2], valid_pieces[:2])
        else:
            cnn.train_model(train_pieces, valid_pieces)
    elif args.action == "test":
        print("Evaluating...")
        if args.onsets_only:
            l2 = evaluate.evaluate_onsets(valid_pieces)
        else:
            l2 = evaluate.evaluate_no_onsets(valid_pieces)
        print("L2 score:", l2)
    elif args.action == "midi":
        print("Transcribing first validation piece to MIDI...")
        output_file = generate_midi(valid_pieces[0], is_WAV=False)
        print("Generated MIDI file:", output_file)