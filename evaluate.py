from constants import *
from util import *
from statistics import harmonic_mean
import cnn
import clean_data
import numpy as np
import pretty_midi
import h5py


# This is a piecewise function for the loss of a note based on it being early/late.
l = lambda distInMs: 0 if distInMs < 20 else distInMs**0.8

# In the evaluation process, detected notes are "searched" for the N-hundred milliseconds before and after the note.
# If the note isn't found, just assume it's missing.
MAX_SEARCH_DIST_IN_MS = 600.0

# The loss on a single note if it's not found within the search distance.
NOT_FOUND_LOSS = 167.0

# Converts CQT time axis to milliseconds
MILLISECONDS_PER_SLICE = 1.0 / SLICE_SAMPLING_RATE * 1000


def generate_midi(filename, is_WAV=False):
	if is_WAV:
		assert(is_valid_file(filename, "wav"))
		cqt = convert_wav_to_cqt(wavPath)
	else:
		assert(is_valid_file(filename, "h5"))
		hf = h5py.File(filename, "r")
		cqt = np.array(hf.get("cqt"))

	# Create a PrettyMIDI object
	pm = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a cello instrument
	piano_program = pretty_midi.instrument_name_to_program('Piano')
	piano = pretty_midi.Instrument(program=piano_program)

	model = load_best_model()
	pianoroll = predicted_pianoroll(model, cqt)
	notes_list = get_notes(pianoroll)

	for note_name, start, end in notes_list:
	    note_number = pretty_midi.note_name_to_number(note_name)
	    note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=end)
	    piano.notes.append(note)

	# Add the piano instrument to the PrettyMIDI object
	pm.instruments.append(piano)
	# Write out the MIDI data
	pm.write('%s.mid' % filename.replace(".h5", ""))


def covert_to_onsets(pianoroll):
	"""
	Converts output pianoroll into just a 1 followed by 0s for onsets.
	"""
    # Iterate through keyboard note by keyboard note
    for row_index in range(pianoroll.shape[1]):
        is_prev_on = False
        # NOTE: Iterate through time slices for just this note's pianoroll
        for slice_index in range(pianoroll.shape[0]):
            if pianoroll[slice_index, row_index]:
                if is_prev_on:
                    pianoroll[slice_index, row_index] = 0
                is_prev_on = True
            else:
                is_prev_on = False


def calculate_percent_loss(pianoroll_1, pianoroll_2, length):
    total_loss = 0.0
    # Calculate max loss by pretending no notes are found
    max_total_loss = 0.0

    for row_index in range(pianoroll_1.shape[1]):
        # Calculate loss independently within each row
        for slice_index_1 in range(pianoroll_1.shape[0]):
            if slice_index_1 == length:
            	break

            is_on = pianoroll_1[slice_index_1, row_index]
            if is_on:
                loss = 0.0
                is_found = False

                # Search right
                slice_index_2 = slice_index_1
                while slice_index_2 < slice_index_1 + MAX_SEARCH_DIST_IN_MS / MILLISECONDS_PER_SLICE:
                    if slice_index_2 >= len(pianoroll_2[row_index]):
                        loss = NOT_FOUND_LOSS
                        break
                    if pianoroll_2[row_index][slice_index_2]:
                        loss = l((slice_index_2 - slice_index_1) * MILLISECONDS_PER_SLICE)
                        is_found = True
                        break
                    slice_index_2 += 1

                if not is_found:
                    # Search left
                    slice_index_2 = slice_index_1 - 1
                    while slice_index_2 > slice_index_1 - MAX_SEARCH_DIST_IN_MS / MILLISECONDS_PER_SLICE:
                        if slice_index_2 < 0:
                            loss = NOT_FOUND_LOSS
                            break
                        if pianoroll_2[row_index][slice_index_2]:
                            # If the note is closer to the left, use that loss
                            loss = min(loss, l((slice_index_1 - slice_index_2) * MILLISECONDS_PER_SLICE))
                            is_found = True
                            break
                        slice_index_2 -= 1

                if not is_found:
                    loss = NOT_FOUND_LOSS

                total_loss += loss
                max_total_loss += NOT_FOUND_LOSS

    return total_loss / max_total_loss


def get_pianorolls(filename):
	"""
	Get the golden and predicted pianorolls given an h5 file
	containing the cqt and pianoroll, as well as their length.
	"""
	assert(is_valid_file(filename, "h5"))
	hf = h5py.File(filename, "r")
	cqt = np.array(hf.get("cqt"))

    # Evaluate on precision and recall
    model = load_best_model()
	predicted = predicted_pianoroll(model, cqt)
	golden = np.array(hf.get("pianoroll"))
    length = min(predicted.shape[0], golden.shape[0])

    return predicted, golden, length


def evaluate_onsets(filenames):
	"""
	Evaluates the accuracy of predictions using our custom function.

	Uses average precision and recall over all files in the test set,
	which implicitly assumes that most pieces have roughly the same size.
	"""
	precisions = []
	recalls = []
	for filename in filenames:
		predicted, golden, length = get_pianorolls(filename)

	    # Precision--iterate over output
	    precision = 1.0 - calculate_percent_loss(predicted, golden, length)

	    # Recall-iterate over golden
	    recall = 1.0 - calculate_percent_loss(golden, predicted, length)

	    precisions.append(precision)
	    recall.append(recall)

    return harmonic_mean((np.mean(precision), np.mean(recall)))


def evaluate_no_onsets(filenames):
	"""
	Evaluates the accuracy of predictions using the percent overlap in pianoroll.

	Keeps track of numerator and denominators separately and uses harmonic_mean
	to distill these down to a single metric.
	"""
	conjunction = 0
	total_predicted = 0
	total_golden = 0
	for filename in filenames:
		predicted, golden, length = get_pianorolls(filename)

		# Trim the pianorolls to the same size (some clipping may have occurred during prediction)
		predicted = predicted[:length, :]
		golden = golden[:length, :]

		covert_to_onsets(predicted)
		covert_to_onsets(golden)

		conjunction += np.sum(np.logical_and(predicted, golden))
		total_predicted += np.sum(predicted)
		total_golden += np.sum(golden)

	precision = conjunction / total_predicted
	recall = conjunction / total_golden

	return harmonic_mean((precision, recall))