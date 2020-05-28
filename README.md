# Machine Music Transcription

Automatic audio transcription from .wav to MIDI

## To get started:

run `python main.py sample-cqt-dir sample-pianoroll-dir`

## File structure

### Scripts
main.py - main program loop
cnn.py - builds, trains, and runs inference on CNN
constants.py - all constants 
evaluateTestPianoroll.py - evaluates pianoroll generated from inference
midiToPianoroll.py - conversions between midi and pianoroll slices
wav_to_cqt_slices.py - conversions between wav and constant-q transform slices
noisifyMidis.py

### Directories
sample-cqt-dir
sample-pianoroll-dir
