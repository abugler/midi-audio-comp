import os
import numpy as np
import torch
import yaml
from yaml import Loader
from torch.utils.data import Dataset
from pretty_midi import PrettyMIDI
import librosa
import soundfile as sf
import nussl
from nussl.datasets import Slakh
from nussl.core.audio_signal import STFTParams
from nussl import AudioSignal
from collections import namedtuple

train = "../slakh_16k/train"
test = "../slakh_16k/test"

stft_params = STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

class PairedSlakh(Dataset):
    def __init__(self, root, stft_params=stft_params,
        excerpt_length=8,
        sample_rate=16_000):
        self.__dict__.update(locals())
        self.items = self.get_items(root)
        self.stft_params = stft_params

    def __len__(self):
        return len(self.items) * 2

    def get_items(self, folder):

        trackpaths = [os.path.join(folder, f) for f in os.listdir(folder)]

        midi_audio = []
        for trackpath in trackpaths:
            with open(os.path.join(trackpath, 'metadata.yaml'), 'r') as file:
                metadata = yaml.load(file, Loader=Loader)
            for stem in metadata["stems"].keys():
                if metadata["stems"][stem]["program_num"] == 128:
                    continue # remove all drums, for now
                audio = os.path.join(trackpath, 'stems', stem + ".wav")
                midi = os.path.join(trackpath, 'MIDI', stem + ".mid")
                if os.path.isfile(audio):
                    midi_audio.append((audio, midi))

        return midi_audio


    def random_midi_clip(self, midi):
        """
        Loads a midi file, randomly clips it, and returns
        the starting second, and the clipped midi file
        """
        pm = PrettyMIDI(midi_file=midi)
        # Randomly select a note to start
        # This ensures that a note will
        # start in the first half of the clip
        start = max(
            np.random.choice(
                pm.instruments[0].notes
            ).start - np.random.random() * .5 * self.excerpt_length,
            0
        )
        end = start + self.excerpt_length
        new_notes = []
        for note in pm.instruments[0].notes:
            if note.end < start or note.start > end:
                continue
            note.end = min(end, note.end) - start
            note.start = max(start, note.start) - start
            new_notes.append(note)
        pm.instruments[0].notes = new_notes
        return pm, start

    def add_signal_padding(self, signal):
        signal.audio_data = np.concatenate(
            (signal.audio_data,
             np.zeros((1,
                max(0, int(self.excerpt_length
                           * self.sample_rate
                           - signal.signal_length))
                ))),
            axis=1
        )


    def to_dict(self, audio, midi, same):
        def get_piano_roll(pm, times): 
            roll = pm.get_piano_roll(fs=1/times[1], times=times)
            roll = (roll > 30).astype(int)
            low = librosa.note_to_midi('A0')
            hi = librosa.note_to_midi('C8') + 1
            roll = roll[low:hi, :]
            return roll
        pm, start = self.random_midi_clip(midi)

        f = sf.SoundFile(audio)

        signal = AudioSignal(
            path_to_input_file=audio,
            stft_params=self.stft_params,
            sample_rate=self.sample_rate,
            offset=start if same
                else max(np.random.uniform(
                    0,
                    len(f) / self.sample_rate - self.excerpt_length), 0),
            duration=self.excerpt_length)
        self.add_signal_padding(signal)
        signal.stft()
        spectrogram = signal.magnitude_spectrogram_data
        times = signal.time_bins_vector
        roll = get_piano_roll(pm, times)

        # To Torch
        spectrogram = torch.tensor(spectrogram[..., 0]).float()
        roll = torch.tensor(roll).float()
        same = torch.tensor(same).float()
        return {
            'spectrogram': spectrogram,
            'piano_roll': roll,
            'same': same
        }

    def __getitem__(self, i):
        """
        If i >= len(self.items), choose a random pair of stem and midi.

        Otherwise, chose the midi/audio pair at self.items[i]
        """
        n = len(self.items)
        if i < n:
            audio, midi = self.items[i]
            same = True
        else:
            i, j = np.random.choice(n, 2, replace=False)
            audio, _ = self.items[i]
            _, midi = self.items[j]
            same = False

        return self.to_dict(audio, midi, same)


