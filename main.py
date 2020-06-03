import os
import re
import collections
import time
import constants
import cnn
import pdb
import argparse

# # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
# from keras import backend
# assert len(backend.tensorflow_backend._get_available_gpus()) > 0

import evaluateTestPianoroll
    
def train(cqt_dir, pianoroll_dir):
    # Train CNN on slices and pianoroll
    
    # TODO: these should be organized cqt_dir > piece_dir > slice.bin
    cqtSlicePaths = [os.path.join(cqt_dir, cqt_slice_path) for cqt_slice_path in os.listdir(cqt_dir)]
    pianoPaths = [os.path.join(pianoroll_dir, pianoroll_path) for pianoroll_path in os.listdir(pianoroll_dir)]
    cnn.run_cnn(cqtSlicePaths, pianoPaths)

    return

def test():
    
    for csvPath in os.listdir(constants.TEST_CQT_CNN_SLICES_IN_DIR):
        csvPath = os.path.join(constants.TEST_CQT_CNN_SLICES_IN_DIR, csvPath).replace("\\", "/")
        pieceID = os.path.basename(csvPath)[:-4]
        pianorollOutputPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, 
                                           constants.PIANOROLL_NAME_TEMPLATE % (pieceID, ""))
        cnn.make_predictions(csvPath, pianorollOutputPath)

    # Compare CNN output with pianoroll and assess accuracy
    # TODO: make sure each file corresponds to a complete output piano roll
    resultsFilename = constants.RESULTS_PATH % time.strftime("%Y%m%d-%H%M%S")
    with open(resultsFilename, "w") as resultsFile:
        # NOTE: Calculate overall accuracy by averaging across the results for the different full pieces.
        avgPrecision = 0.0
        avgRecall = 0.0
        denominator = 0
        for outputPianorollPathFilename in os.listdir(constants.TEST_PIANOROLL_OUT_DIR):
            
            outputPianorollPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, outputPianorollPathFilename)
            goldenPianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, outputPianorollPathFilename)

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

def main():
    train()
    print("Training complete!")
    return

    test()
    print("Testing complete! See numericalResults.txt for results!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Trains a CNN.')
    parser.add_argument('cqt_dir', help='Directory of CQT slices')
    parser.add_argument('pianoroll_dir', help='Directory of corresponding pianorolls')
    args = parser.parse_args()
    train(args.cqt_dir, args.pianoroll_dir)
