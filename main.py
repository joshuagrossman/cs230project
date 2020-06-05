import os
import re
import collections
import time
from constants import *
import cnn
import pdb
import argparse
import h5py
import math

# # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
# from keras import backend
# assert len(backend.tensorflow_backend._get_available_gpus()) > 0

import evaluateTestPianoroll


def train(data_dir):
    # Train CNN on slices and pianoroll

    model, history = cnn.train_model(train_pieces)

    return model, history


def test(test_pieces, model, history):
    
    for csvPath in os.listdir(TEST_CQT_CNN_SLICES_IN_DIR):
        csvPath = os.path.join(TEST_CQT_CNN_SLICES_IN_DIR, csvPath).replace("\\", "/")
        pieceID = os.path.basename(csvPath)[:-4]
        pianorollOutputPath = os.path.join(TEST_PIANOROLL_OUT_DIR, 
                                           PIANOROLL_NAME_TEMPLATE % (pieceID, ""))
        cnn.make_predictions(csvPath, pianorollOutputPath)

    # Compare CNN output with pianoroll and assess accuracy
    # TODO: make sure each file corresponds to a complete output piano roll
    resultsFilename = RESULTS_PATH % time.strftime("%Y%m%d-%H%M%S")
    with open(resultsFilename, "w") as resultsFile:
        # NOTE: Calculate overall accuracy by averaging across the results for the different full pieces.
        avgPrecision = 0.0
        avgRecall = 0.0
        denominator = 0
        for outputPianorollPathFilename in os.listdir(TEST_PIANOROLL_OUT_DIR):
            
            outputPianorollPath = os.path.join(TEST_PIANOROLL_OUT_DIR, outputPianorollPathFilename)
            goldenPianorollPath = os.path.join(TEST_PIANOROLL_GOLDEN_DIR, outputPianorollPathFilename)

            precision, recall = evaluateTestPianoroll.evaluate(outputPianorollPath, goldenPianorollPath)
            avgPrecision += precision
            avgRecall += recall
            denominator += 1
            resultsFile.write(outputPianorollPath + ":" + "\n")
            resultsFile.write("\tPrecision: " + str(precision) + "\n")
            resultsFile.write("\tRecall: " + str(recall) + "\n")

        avgPrecision /= denominator
        avgRecall /= denominator
        resultsFile.write("Overall:" + "\n")
        resultsFile.write("Precision: " + str(avgPrecision) + "\n")
        resultsFile.write("Recall: " + str(avgRecall) + "\n")
    print("Evaluation complete! See numericalResults.txt for results.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Trains a CNN.')
    parser.add_argument('data_dir', help='Directory of h5 files')
    args = parser.parse_args()

    # Get h5 files corresponding to pieces
    piece_paths = [os.path.join(args.data_dir, h5_file) \
        for h5_file in os.listdir(args.data_dir) if "Chamber" not in h5_file]

    # Split pieces into train/test
    boundary = math.floor((1.0 - TEST_SPLIT) * len(piece_paths))
    train_pieces = piece_paths[:boundary]
    test_pieces = piece_paths[boundary:]

    model, history = train(train_pieces)
    # test(test_pieces, model, history)