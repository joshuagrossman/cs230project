# Machine Music Transcription

Automatic audio transcription from .wav to MIDI

## To get started:

To train, run `python main.py <processed h5 file directory> train`

To get a sample MIDI transcription, run `python main.py <processed h5 file directory> midi`

The name of the generated file is specified in the console output, which you can scp from your local machine and play back with your default audio player.

## File structure

### Scripts
main.py - main program loop

cnn.py - specifies CNN model

train.py - trains the CNN

evaluate.py - functions related to evaluation

constants.py - all constants 

util.py - utility functions

evaluateTestPianoroll.py - evaluates pianoroll generated from inference

wav_to_cqt_slices.py - conversions between wav and constant-q transform slices

noisifyMidis.py

### Directories
sample-cqts

sample-pianorolls
