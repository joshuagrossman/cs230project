# Machine Music Transcription

Automatic audio transcription from .wav to MIDI

## To get started:

Download the data at: LINK TBD

run `python main.py`

## File structure

### Scripts
cnn.py - builds, trains, and runs inference on CNN
constants.py - all constants 
evaluateTestPianoroll.py - evaluates pianoroll generated from inference
main.py - main program loop
midiToPianoroll.py - conversions between midi and pianoroll
noisifyMidis.py
timeSeriesToCqtSlice.py
wavToTimeSeries.py

### Directories
Train-CQT-CNN-Slices-In
Train-Pianoroll-Out
Train-Time-Series-In
Test-Pianoroll-Golden
Test-Pianoroll-Out
Test-Time-Series-In
Models
WTC1-MIDI
WTC1-WAV
WTC2-MIDI
WTC2-WAV
