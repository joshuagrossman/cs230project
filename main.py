from constants import *
from util import *
import evaluate
import train
import os
import re
import time
import pdb
import argparse
import h5py
import random


def get_train_valid_test_split(data_dir):
    # Get h5 files corresponding to pieces
    piece_paths = [os.path.join(data_dir, file) \
        for file in os.listdir(data_dir) \
        if "Chamber" not in file and is_valid_file(file, "h5")]

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
    parser.add_argument('model_ckpt_dir', help='The directory containing the model weights you want to use')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, dest='lr')
    parser.add_argument('--n-epochs', type=int, default=NUM_EPOCHS, dest='n_epochs')
    parser.add_argument('--resume', action='store_const', const=True, dest='resume',
                        help='Resume training from last checkpoint in model_ckpt_dir')
    parser.add_argument('--quick-train', action='store_const', const=False, dest='quick_train',
                        help='Use only two pieces for training and validation')
    parser.add_argument('--onsets-only', action='store_const', const=False, dest='onsets_only',
                        help='Whether to use onsets_only for testing')
    parser.add_argument('--test-set', action='store_const', const=False, dest='test_set',
                        help='Whether to evaluate the test set')
    args = parser.parse_args()

    # Input validation
    if not os.path.isdir(args.model_ckpt_dir):
        if args.resume:
            raise Exception("Attempting to resume training from nonexistant weights directory: %s" \
                % args.model_ckpt_dir)
        else:
            print("The specified weights directory %s doesn't exist. Creating it now." \
                % args.model_ckpt_dir)
            os.system("mkdir %s" % args.model_ckpt_dir)
    if args.lr <= 0 or args.lr >= 1:
        raise Exception("Invalid learning rate %f" % args.lr)
    if args.n_epochs <= 0 or args.n_epochs >= 1000:
        raise Exception("Invalid number of epochs %f. Expected an integer between 1 - 999." % args.n_epochs)
    if not os.path.isdir(args.data_dir) \
        or not any([is_valid_file(file, "h5") for file in os.listdir(args.data_dir)]):
        raise Exception("Specified data directory is empty or doesn't exist: %s" % args.data_dir)

    # Get train/dev/test split
    train_pieces, valid_pieces, test_pieces = get_train_valid_test_split(args.data_dir)

    # Command line argument logic
    if args.action == "train":
        if args.quick_train:
            train.train_model(train_pieces[:2],
                              valid_pieces[:2],
                              batch_size=8,
                              n_epochs=args.n_epochs,
                              lr=args.lr,
                              model_ckpt_dir=args.model_ckpt_dir,
                              resume=args.resume)
        else:
            train.train_model(train_pieces,
                              valid_pieces,
                              n_epochs=args.n_epochs,
                              lr=args.lr,
                              model_ckpt_dir=args.model_ckpt_dir,
                              resume=args.resume)
    elif args.action == "test":
        print("Evaluating...")
        evaluation_set = test_pieces if args.test_set else valid_pieces
        if args.onsets_only:
            l2 = evaluate.evaluate_onsets(evaluation_set, models_dir=args.model_ckpt_dir)
        else:
            l2 = evaluate.evaluate_no_onsets(evaluation_set, models_dir=args.model_ckpt_dir)
        print("L2 score:", l2)
    elif args.action == "midi":
        print("Transcribing first validation piece to MIDI...")
        output_file = evaluate.generate_midi(valid_pieces[0], is_WAV=False, models_dir=args.model_ckpt_dir)
        print("Generated MIDI file:", output_file)
