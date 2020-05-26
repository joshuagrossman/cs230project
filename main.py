import os
import re
import collections
import time
import constants
import cnn
import wavToTimeSeries
import timeSeriesToCqtSlice
import midiToPianoroll
import pdb

# # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
# from keras import backend
# assert len(backend.tensorflow_backend._get_available_gpus()) > 0

import evaluateTestPianoroll

def train():
    # # Do the same conversion for each file in WTC1
    # # fileSearchKeys = [(constants.PRELUDE_WAV_NAME_TEMPLATE % x) for x in range(1, 25)] + \
    # #                  [(constants.FUGUE_WAV_NAME_TEMPLATE % x) for x in range(1, 25)]

    # fileSearchKeys = [(constants.FUGUE_WAV_NAME_TEMPLATE % str(x)) for x in range(1, 25)]

    # # To be populated
    # # pieceIDs = []
    # # wavPaths = []
    # # csvPaths = []
    # # cqtSlicePaths = []
    # # pianoPaths = []
    
    # # Convert all wav to time series if time series don't exist
    # # for fileSearchKey in fileSearchKeys:
    # #     # This should be TRAIN_TIME_SERIES_IN_DIR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # #     for wavFilename in os.listdir(constants.WTC1_WAV_DIR):
    # #         match = re.search(fileSearchKey, wavFilename)
    # #         if match:
    # #             pieceID = match.group(1)
    # #             # pieceIDs.append(pieceID)
    # #             wavPath = os.path.join(constants.WTC1_WAV_DIR, wavFilename).replace("\\", "/")
    # #             csvPath = constants.TRAIN_TIME_SERIES_IN_DIR + "/" + pieceID + ".csv"
    # #             # wavPaths.append(wavPath)
    # #             # csvPaths.append(csvPath)
    # #             wavToTimeSeries.convert(wavPath, csvPath)

    # # print "Time series data generated for training!\n"

    # # Convert all time series to CQT slices if slices don't exist
    # for csvPath in os.listdir(constants.TRAIN_TIME_SERIES_IN_DIR):
    #     csvPath = os.path.join(constants.TRAIN_TIME_SERIES_IN_DIR, csvPath).replace("\\", "/")
    #     pieceID = os.path.basename(csvPath)[:-4]
    #     # pieceIDs.append(pieceID)
    #     cqtSliceDir = constants.TRAIN_CQT_CNN_SLICES_IN_DIR
    #     curCqtSlicePaths = timeSeriesToCqtSlice.convert(csvPath, cqtSliceDir, pieceID)
    #     # cqtSlicePaths.extend(curCqtSlicePaths)
    # print "CQT slices generated for training!\n"

    # # Convert all MIDI to pianoroll labeled the same as CQT slices
    # pianorollIndicesToSample = collections.defaultdict(list)
    # # Construct dict of pieceID -> list of ms offsets

    # for cqtSlicePath in os.listdir(constants.TRAIN_CQT_CNN_SLICES_IN_DIR):
    #     cqtSlicePath = os.path.join(constants.TRAIN_CQT_CNN_SLICES_IN_DIR, cqtSlicePath).replace("\\", "/")
    #     matchPrelude = re.search(constants.FUGUE_WAV_NAME_TEMPLATE % "[0-9]+", cqtSlicePath)
    #     matchFugue = re.search(constants.PRELUDE_WAV_NAME_TEMPLATE % "[0-9]+", cqtSlicePath)
    #     pieceID = matchPrelude.group(1) if matchPrelude else matchFugue.group(1)
    #     match = re.search(constants.CQT_SLICE_NAME_TEMPLATE % (pieceID, "[0-9]+"), cqtSlicePath)
    #     sliceTimeInMs = int(match.group(1))
    #     pianorollIndicesToSample[pieceID].append(sliceTimeInMs)

    # # for cqtSlicePath in os.listdir(constants.TRAIN_CQT_CNN_SLICES_IN_DIR):
    # #     cqtSlicePath = cqtSlicePath.replace("\\", "/")
    # #     for pieceID in pieceIDs:
    # #         match = re.search(constants.CQT_SLICE_NAME_TEMPLATE % (pieceID, "[0-9]+"), cqtSlicePath)
    # #         if match:
    # #             sliceTimeInMs = int(match.group(1))
    # #             pianorollIndicesToSample[pieceID].append(sliceTimeInMs)

    # # Get the relevant pianoroll slices

    # for pieceID in pianorollIndicesToSample:
    #     midiFilename = os.path.join(constants.WTC1_MIDI_DIR, (constants.MIDI_OUT_NAME_TEMPLATE % pieceID)).replace("\\", "/")
    #     pianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiFilename)
    #     for sliceTimeInMs in pianorollIndicesToSample[pieceID]:
    #         pianorollSlice = midiToPianoroll.get_ground_truth_labels_at_time(pianoroll, sliceTimeInMs)
    #         if type(pianorollSlice) == bool:
    #             continue
    #         pianorollSlicePath = os.path.join(constants.TRAIN_PIANOROLL_OUT_DIR, constants.PIANOROLL_NAME_TEMPLATE % (pieceID, sliceTimeInMs)).replace("\\", "/").replace("(", "").replace(")", "")
    #         midiToPianoroll.writePianorollToFile(pianorollSlice, pianorollSlicePath)
    #         # pianoPaths.append(pianorollSlicePath)
    # print "MIDI pianoroll generated for training!\n"
    # return

    cqtSlicePaths = [os.path.join(constants.TRAIN_CQT_CNN_SLICES_IN_DIR, p).replace("\\", "/") for p in os.listdir(constants.TRAIN_CQT_CNN_SLICES_IN_DIR)]
    pianoPaths = [os.path.join(constants.TRAIN_PIANOROLL_OUT_DIR, p).replace("\\", "/") for p in os.listdir(constants.TRAIN_PIANOROLL_OUT_DIR)]

    # Train CNN on slices and pianoroll
    cnn.run_cnn(cqtSlicePaths, pianoPaths)

    cqtSlicePaths = [os.path.join(constants.TRAIN_CQT_CNN_SLICES_IN_DIR, p).replace("\\", "/") for p in os.listdir(constants.TRAIN_CQT_CNN_SLICES_IN_DIR)]
    pianoPaths = [os.path.join(constants.TRAIN_PIANOROLL_OUT_DIR, p).replace("\\", "/") for p in os.listdir(constants.TRAIN_PIANOROLL_OUT_DIR)]

    # return

def test():
    
    fileSearchKeys = [(constants.PRELUDE_WAV_NAME_TEMPLATE % str(x)) for x in range(3, 25)]

#     # Convert all wav to time series if time series don't exist
#     for wavFilename in os.listdir(constants.WTC2_WAV_DIR):
#         pieceID = os.path.basename(wavFilename)[:-4]
#         wavPath = os.path.join(constants.WTC2_WAV_DIR, wavFilename).replace("\\", "/")
#         csvPath = os.path.join(constants.TEST_TIME_SERIES_IN_DIR, pieceID + ".csv").replace("\\", "/")
#         wavToTimeSeries.convert(wavPath, csvPath)
#     print("Time series data generated for testing!\n")
#     return
    # To be populated
    # pieceIDs = []
    # wavPaths = []
    # csvPaths = []
    # cqtSlicePaths = []
    
    # Convert all time series to CQT slices
    for csvPath in os.listdir(constants.TEST_TIME_SERIES_IN_DIR):
        csvPath = os.path.join(constants.TEST_TIME_SERIES_IN_DIR, csvPath).replace("\\", "/")
        pieceID = os.path.basename(csvPath)[:-4]
        cqtSliceDir = constants.TEST_CQT_CNN_SLICES_IN_DIR
        cqtData = timeSeriesToCqtSlice.convertTest(csvPath, cqtSliceDir, pieceID)

        pianorollOutputPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("\\", "/").replace("(", "").replace(")", "")

        cnn.make_predictions(cqtData, pianorollOutputPath)

        midiFilename = pieceID + ".mid"
        midiPath = os.path.join(constants.WTC2_MIDI_DIR, midiFilename).replace("\\", "/")
        pianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("\\", "/").replace("(", "").replace(")", "")
        fullPianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiPath, constants.CQT_SAMPLING_RATE)
        midiToPianoroll.writePianorollToFile(fullPianoroll, pianorollPath)

        if len(cqtData[0]) != len(fullPianoroll[0]):
            print("CQT and pianoroll have different length: ", str(len(cqtData[0])), str(len(fullPianoroll[0])))

        print(pieceID + " has been transcribed.")

    print("Done transcribing. Now evaluating.\n")
    # print("CQT slices generated for testing!\n")
    
    # Convert all MIDI to complete pianoroll
    # for midiPath in os.listdir(constants.WTC2_MIDI_DIR):
    #     midiPath = os.path.join(constants.WTC2_MIDI_DIR, midiPath).replace("\\", "/")
    #     matchPrelude = re.search(constants.FUGUE_WAV_NAME_TEMPLATE % "[0-9]+", midiPath)
    #     matchFugue = re.search(constants.PRELUDE_WAV_NAME_TEMPLATE % "[0-9]+", midiPath)
    #     pieceID = matchPrelude.group(1) if matchPrelude else matchFugue.group(1)
    #     pianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("\\", "/").replace("(", "").replace(")", "")
    #     fullPianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiPath, constants.CQT_SAMPLING_RATE)
    #     midiToPianoroll.writePianorollToFile(fullPianoroll, pianorollPath)

    # for pieceID in pieceIDs:
    #     midiPath = os.path.join(constants.WTC2_MIDI_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/")
    #     pianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("\\", "/").replace("(", "").replace(")", "")
    #     fullPianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiPath, constants.CQT_SAMPLING_RATE)
    #     midiToPianoroll.writePianorollToFile(fullPianoroll, pianorollPath)
    # print "Generated golden pianoroll files!\n"

    # cqtSlicePaths = [os.path.join(constants.TEST_CQT_CNN_SLICES_IN_DIR, p).replace("\\", "/") for p in os.listdir(constants.TEST_CQT_CNN_SLICES_IN_DIR)]
    # output filename should look be aggregated into Fugue_No._1_Pianoroll_.csv

    # Compare CNN output with pianoroll and assess accuracy
    # TODO: make sure each file corresponds to a complete output piano roll
    resultsFilename = constants.RESULTS_PATH % time.strftime("%Y%m%d-%H%M%S")
    with open(resultsFilename, "w") as resultsFile:
        avgPrecision = 0.0
        avgRecall = 0.0
        denominator = 0
        for outputPianorollPathFilename in os.listdir(constants.TEST_PIANOROLL_OUT_DIR):
            outputPianorollPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, outputPianorollPathFilename).replace("\\", "/").replace("(", "").replace(")", "")
            goldenPianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, outputPianorollPathFilename).replace("\\", "/").replace("(", "").replace(")", "")
            # pianorollFilename = (constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("(", "").replace(")", "")
            # outputPianorollPath = os.path.join(constants.TEST_PIANOROLL_OUT_DIR, pianorollFilename).replace("\\", "/")
            # goldenPianorollPath = os.path.join(constants.TEST_PIANOROLL_GOLDEN_DIR, pianorollFilename).replace("\\", "/")
            precision, recall = evaluateTestPianoroll.evaluate(outputPianorollPath, goldenPianorollPath)
            avgPrecision += precision
            avgRecall += recall
            denominator += 1
            resultsFile.write(outputPianorollPath + ":" + "\n")
            resultsFile.write("\tPrecision: " + str(precision) + "\n")
            resultsFile.write("\tRecall: " + str(recall) + "\n")
            # resultsFile.write("Listen shortly at " + os.path.join(constants.TEST_MIDI_OUT_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/") + "\n")

        avgPrecision /= denominator
        avgRecall /= denominator
        resultsFile.write("Overall:" + "\n")
        resultsFile.write("Precision: " + str(avgPrecision) + "\n")
        resultsFile.write("Recall: " + str(avgRecall) + "\n")
    print("Evaluation complete! See numericalResults.txt for results.\n")

    # Generate MIDIs from pianoroll for human listening
    # for pieceID in pieceIDs:
    #     pianorollFilename = (constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("(", "").replace(")", "")
    #     outputPianorollPath = (constants.TEST_PIANOROLL_OUT_DIR + pianorollFilename).replace("\\", "/")
    #     midiPath = os.path.join(constants.TEST_MIDI_OUT_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/")
    #     pianoroll = midiToPianoroll.format_piano_roll_for_midi(outputPianorollPath)
    #     prettyMidi = midiToPianoroll.piano_roll_to_pretty_midi(pianoroll, constants.CQT_SAMPLING_RATE)
    #     prettyMidi.write(midiPath)
    # print("MIDIs generated for human listening!\n" 

def main():
#     train()
#     print("Training complete!")
#     return

    test()
    print("Testing complete! See numericalResults.txt for results!")

if __name__ == '__main__':
    main()
