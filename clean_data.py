import sys
import argparse
import numpy as np
import pretty_midi
import librosa
from constants import *
import pdb
import collections
import os
import re
import multiprocessing as mp
import time
import h5py
import os.path
import math

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

def convert_wav_to_cqt(wavPath):
    """
    Converts a WAV file to a CQT representation
    """
    y, sr = librosa.load(path=wavPath, sr=None, offset=0)
    
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=CONTEXT_WINDOW_ROWS, 
                           bins_per_octave=BINS_PER_OCTAVE, 
                           hop_length=HOP_LENGTH, 
                           filter_scale=FILTER_SCALE))
    return cqt

def get_piano_roll_subdivided_by_ms(midi_path, sampling_rate=SAMPLE_RATE_IN_HZ):
    """
    Returns an np array that is a piano roll representation of a midi file, cropping the
    128 note values in MIDI to the 88 notes of a piano keyboard.
    """
    
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    raw_piano_roll = midi_data.get_piano_roll(fs=sampling_rate, times=None)
    piano_roll = raw_piano_roll > 0
    # 21 extra at bottom of range (C-1 -> G#0), 19 extra at top (C#8 -> G9)
    # TODO: look at piano check to make sure we're removing the 
    # right one (not flipped top/bottom)... else [21:-19]
    return np.asarray(piano_roll[19:-21]).astype(int) 

def get_cqt_and_pianoroll(wav_path, midi_dir, output_dir):
    """
    Converts a WAV file into CQT representation with corresponding MIDI pianoroll.
    """
    
    if not is_valid_file(wav_path, "wav"):
        print("Invalid WAV file: " + wav_path)
        return
    
    piece_id = os.path.basename(wav_path).replace(".wav", "")
    
    midi_path = os.path.join(midi_dir, piece_id) + ".midi"
    
    if not os.path.exists(midi_path):
        print("Could not locate MIDI file: " + midi_path)
        return
    
    print("Converting to CQT and extracting pianorolls for " + piece_id)
    
    cqt = convert_wav_to_cqt(wav_path)
    pianoroll = get_piano_roll_subdivided_by_ms(midi_path)
    
    print("Saving CQT and pianorolls for " + piece_id)
    
    tot_time = cqt.shape[1]
    radius = CQT_SLICE_RADIUS_IN_PIXELS
    offset = CQT_SLICE_OFFSET_IN_PIXELS
    
    onsets = np.array(range(radius, tot_time-offset-radius-1, 2*radius+1)) 
    slice_indices = np.array([(onset+offset-radius, onset+offset+radius+1) for onset in onsets])
    h5_name = os.path.join(output_dir, piece_id) + ".h5"
    
    with h5py.File(h5_name, 'w') as hf:
        hf.create_dataset("onsets", data=onsets)
        hf.create_dataset("slice_indices", data=slice_indices)
        hf.create_dataset("cqt", data=cqt)
        hf.create_dataset("pianoroll", data=pianoroll)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts a directory of WAV files to CQT Slices and grabs corresponding pianorolls')
    parser.add_argument('wav_dir', help='Input directory of WAV files')
    parser.add_argument('midi_dir', help='Input directory of MIDI files')
    parser.add_argument('output_dir', help='Output directory of CQT Slices and Pianorolls')
    args = parser.parse_args()
    
    if not os.path.exists(args.wav_dir):
        print("Invalid WAV directory. Stopping.")
        sys.exit()
        
    if not os.path.exists(args.midi_dir):
        print("Invalid MIDI directory. Stopping.")
        sys.exit()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    wav_paths = [os.path.join(args.wav_dir, wav_file) for wav_file in os.listdir(args.wav_dir)]
    
    start_time = time.time()
    
    # TURN DOWN CPU COUNT ON SHARED CLUSTERS -- mp.cpu_count() % 2
    with mp.Pool(mp.cpu_count()) as pool:
        processes = [pool.apply_async(get_cqt_and_pianoroll, args=(wav_path, args.midi_dir, args.output_dir)) for wav_path in wav_paths]
        [process.get() for process in processes]
    
    print(str(round((time.time() - start_time)/60)) + " minutes to convert " + str(len(wav_paths)) + " WAV files to CQT slices and extract corresponding pianorolls.")