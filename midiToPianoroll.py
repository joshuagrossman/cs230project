import sys
import argparse
import numpy as np
import pretty_midi
import librosa
import constants
import pdb
import collections
import os
import re
import multiprocessing as mp
import time
import h5py

def is_valid_file(filepath, extension):
    """
    Checks if a file is a valid binary file.
    """
    file_re = extension
    
    if extension=="bin":
        file_re = re.compile("\.bin$")
    
    if extension=="wav":
        file_re = re.compile("\.wav$")
        
    if extension=="midi":
        file_re = re.compile("\.midi$")
        
    if extension=="h5":
        file_re = re.compile("\.h5$")
    
    if not file_re.search(filepath):
        print("Invalid " + extension + " file: " + filepath)
        return False
    return True

def get_piano_roll_subdivided_by_ms(midi_path, sampling_rate=constants.SAMPLE_RATE_IN_HZ):
    """
    Returns an np array that is a piano roll representation of a midi file, cropping the
    128 note values in MIDI to the 88 notes of a piano keyboard.
    """
    
    if not is_valid_file(midi_path, "midi"):
        return None
    
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    raw_piano_roll = midi_data.get_piano_roll(fs=sampling_rate, times=None)
    piano_roll = raw_piano_roll > 0
    # 21 extra at bottom of range (C-1 -> G#0), 19 extra at top (C#8 -> G9)
    # TODO: look at piano check to make sure we're removing the 
    # right one (not flipped top/bottom)... else [21:-19]
    return np.asarray(piano_roll[19:-21]).astype(int) 
    
def get_start_dict(cqt_dir):
    """
    Returns a dictionary containing the list of slice start times for each piece.
    """
    
    start_dict = {}
    for cqt_file in os.listdir(cqt_dir):
        cqt_path = os.path.join(cqt_dir, cqt_file)
        if not is_valid_file(cqt_path, "h5"):
            continue
        piece_id = cqt_file.replace(".h5", "")
        with h5py.File(cqt_path, 'r') as hf:
            start_dict[piece_id] = [int(k) for k in list(hf.keys())]
    
    return start_dict
    
    
def convert_midi_to_pianoroll(start_dict, midi_path, pianoroll_dir):
    """
    Writes binary files of the piano roll corresponding to each CQT slice.
    """
    
    piece_id = os.path.basename(midi_path).replace(".midi", "")
    starts = start_dict.get(piece_id)
    
    if starts is None:
        print("Could not locate slice start times for " + piece_id)
        return
    
    pianoroll = get_piano_roll_subdivided_by_ms(midi_path)
    if pianoroll is None:
        print("Could not get pianoroll for " + piece_id)
        return
    
    if not os.path.exists(pianoroll_dir):
        os.makedirs(pianoroll_dir)
        
    print("Saving pianorolls for " + os.path.basename(midi_path))
    
    h5_name = os.path.join(pianoroll_dir, piece_id) + ".h5"
    
    with h5py.File(h5_name, 'w') as hf:
        for sliceTimeInMs in starts:
            pianorollSlice = pianoroll[:,sliceTimeInMs]
            hf.create_dataset(str(sliceTimeInMs), data=pianorollSlice)
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts a directory of MIDI to Pianorolls')
    parser.add_argument('cqt_dir', help='Directory of CQT Slices to Match')
    parser.add_argument('midi_dir', help='Input directory of MIDIs')
    parser.add_argument('pianoroll_dir', help='Output directory of Pianorolls')
    args = parser.parse_args()
    
    midi_paths = [os.path.join(args.midi_dir, midi_file) for midi_file in os.listdir(args.midi_dir)]
    start_dict = get_start_dict(args.cqt_dir)
    
    start_time = time.time()
    
    # TURN DOWN CPU COUNT AS NECESSARY
    with mp.Pool(mp.cpu_count()) as pool:
        processes = [pool.apply_async(convert_midi_to_pianoroll, args=(start_dict, midi_path, args.pianoroll_dir)) for midi_path in midi_paths]
        [process.get() for process in processes]
        
    print(str(round(time.time() - start_time)) + " seconds to convert " + str(len(midi_paths)) + " MIDI files to pianorolls.")

        
        
        
        
        
        
        
        
        
        
# def get_ground_truth_labels_at_time(piano_roll, time):
#     """
#     Takes in a piano roll (created with 1000Hz sampling rate) representation of a MIDI 
#     and a time in milliseconds and returns a (88, 1) vector of ground truth labels.
#     """
#     # NOTE: The result is a vector of 1s and 0s
#     if time >= len(piano_roll[0]):
#         return False
#     return piano_roll[:,time]


# def get_test_piano_roll(midi_filename):
#     """
#     Returns a np array that is a golden piano roll representation of a midi, divided into 1/87s slices
#     """
#     return get_piano_roll_subdivided_by_ms(midi_filename, constants.CQT_SAMPLE_RATE)

# def get_ms_offsets(cqt_dir):
#     """
#     Stores the offset of each slice for each piece in a dictionary.
#     """
    
#     # Construct dict of pieceID -> list of ms offsets
#     pianorollIndicesToSample = collections.defaultdict(list)
#     for cqtSlicePath in os.listdir(cqt_subdir):
        
#         # ignore hidden files and non-csv
#         if (cqtSlicePath[0] == "." or cqtSlicePath[-4:] != ".bin" ):
#             print("Could not read " + cqtSlicePath)
#             continue
            
#         cqtSlicePath = os.path.join(cqt_dir, cqtSlicePath)
#         p = re.compile("\/(.+)\_([0-9]+)\.bin$")
#         file_match = p.search(cqtSlicePath)
#         try:
#             pieceID = file_match.group(1)
#             sliceTimeInMs = int(file_match.group(2))
#             pianorollIndicesToSample[pieceID].append(sliceTimeInMs)
#         except Exception as e:
#             print(e)
#             continue


    
# #### Export MIDI as audio

#     # Generate MIDIs from pianoroll for human listening
#     # for pieceID in pieceIDs:
#     #     pianorollFilename = (constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("(", "").replace(")", "")
#     #     outputPianorollPath = (constants.TEST_PIANOROLL_OUT_DIR + pianorollFilename).replace("\\", "/")
#     #     midiPath = os.path.join(constants.TEST_MIDI_OUT_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/")
#     #     pianoroll = midiToPianoroll.format_piano_roll_for_midi(outputPianorollPath)
#     #     prettyMidi = midiToPianoroll.piano_roll_to_pretty_midi(pianoroll, constants.CQT_SAMPLING_RATE)
#     #     prettyMidi.write(midiPath)
#     # print("MIDIs generated for human listening!\n" 

# ### LOOK AT THIS
# def format_piano_roll_for_midi(piano_roll_path):
#     """
#     Takes in path to piano roll csv and reads csv into numpy array. Adds 19 rows pertaining 
#     to the highest values and 21 rows pertaining to the lowest values to the piano roll.
#     """
#     # NOTE: This is used to prepare a pianoroll matrix for conversion back into a midi, for listening.
#     small_piano_roll = np.genfromtxt(piano_roll_path, delimiter='\t')
#     piano_roll_length = small_piano_roll.shape[1]
#     piano_roll = np.insert(small_piano_roll, 0, np.zeros((19,piano_roll_length)), axis=0)
#     piano_roll = np.insert(piano_roll, piano_roll.shape[0], np.zeros((21,piano_roll_length)), axis=0)
#     return piano_roll.astype(int)

# ## LOOK AT THIS
# def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0): 
#     """
#     Convert a Piano Roll array into a PrettyMidi object
#     with a single instrument.

#     From https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py

#     Parameters
#     ----------
#     piano_roll : np.ndarray, shape=(128,frames), dtype=int
#         Piano roll of one instrument
#     fs : int
#         Sampling frequency of the columns, i.e. each column is spaced apart
#         by ``1./fs`` seconds.
#     program : int
#         The program number of the instrument.
#     Returns
#     -------
#     midi_object : pretty_midi.PrettyMIDI
#         A pretty_midi.PrettyMIDI class instance describing
#         the piano roll.
#     """
#     notes, frames = piano_roll.shape
#     pm = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=program)

#     # pad 1 column of zeros so we can acknowledge inital and ending events
#     piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

#     # use changes in velocities to find note on / note off events
#     velocity_changes = np.nonzero(np.diff(piano_roll).T)

#     # keep track on velocities and note on times
#     prev_velocities = np.zeros(notes, dtype=int)
#     note_on_time = np.zeros(notes)

#     for time, note in zip(*velocity_changes):
#         # use time + 1 because of padding above
#         velocity = piano_roll[note, time + 1]
#         time = time / fs
#         if velocity > 0:
#             if prev_velocities[note] == 0:
#                 note_on_time[note] = time
#                 prev_velocities[note] = velocity
#         else:
#             pm_note = pretty_midi.Note(
#                 velocity=prev_velocities[note],
#                 pitch=note,
#                 start=note_on_time[note],
#                 end=time)
#             instrument.notes.append(pm_note)
#             prev_velocities[note] = 0
#     pm.instruments.append(instrument)
#     return pm

# #### Export MIDI as audio

#     # Generate MIDIs from pianoroll for human listening
#     # for pieceID in pieceIDs:
#     #     pianorollFilename = (constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("(", "").replace(")", "")
#     #     outputPianorollPath = (constants.TEST_PIANOROLL_OUT_DIR + pianorollFilename).replace("\\", "/")
#     #     midiPath = os.path.join(constants.TEST_MIDI_OUT_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/")
#     #     pianoroll = midiToPianoroll.format_piano_roll_for_midi(outputPianorollPath)
#     #     prettyMidi = midiToPianoroll.piano_roll_to_pretty_midi(pianoroll, constants.CQT_SAMPLING_RATE)
#     #     prettyMidi.write(midiPath)
#     # print("MIDIs generated for human listening!\n" 
