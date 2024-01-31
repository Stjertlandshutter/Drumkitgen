import abc
import functools
import itertools
import math
import os
import warnings
import glob
import random
from abc import ABC
from pathlib import Path
from typing import *

import av
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

def get_duration_sec(file, cache=False):
    try:
        with open(file + ".dur", "r") as f:
            duration = float(f.readline().strip("\n"))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration

def load_audio(file, sr, duration, resample=True, approx=False, time_base="samples", check_duration=True):
    resampler = None
    if time_base == "sec":
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    if not os.path.exists(file):
        return np.zeros((2, duration), dtype=np.float32), sr
    
    container = av.open(file)
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    
    ## duration: length for model
    ## audio_duration * sr: real thing 
     
    if(duration <= audio_duration * sr):
        print(f"End {duration} beyond duration {audio_duration*sr}")
    
    if resample:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    else:
        assert sr == audio.sample_rate
    

    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(0, stream=audio)
    total_read = 0
    
    for frame in container.decode(audio=0):  # Only first audio stream
        ## frame = 실제 음원
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)[0]
        frame = frame.to_ndarray(format="fltp")  # Convert to floats and not int16
        read = frame.shape[-1]
        
        if total_read + read > duration:
            read = duration - total_read
        
        ## 알아서 잘라주네 
        sig[:, total_read : total_read + read] = frame[:, :read]
        
        total_read += read
        if total_read == duration:
            break
    
    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"
    
    return sig, sr

def _identity(x):
  return x

class MultiSourceDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, sample_length, audio_files_dir, stems, transform=None):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        self.audio_files_dir = audio_files_dir
        self.stems = stems
        assert (
                sample_length / sr < self.min_duration
        ), f"Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}"
        self.aug_shift = aug_shift
        self.transform = transform if transform is not None else _identity
        self.init_dataset()



    def filter(self, tracks):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])
            
            # skip if there are no sources per track
            if not files:
                continue
            
            durations_track = np.array([get_duration_sec(file, cache=True) * self.sr for file in files]) # Could be approximate
            
            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.sr >= self.max_duration).any():
                continue
            
            keep.append(track)
            durations.append(durations_track[0])
        
        print(f"self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        self.kits = keep
        self.durations = durations
        self.cumsum = np.cumsum(np.array(self.durations))

    def init_dataset(self):
        # Load list of tracks and starts/durations
        tracks = os.listdir(self.audio_files_dir)
        print(f"Found {len(tracks)} kits.")
        self.filter(tracks)


    def get_kit_chunk_randomness(self,index):
        kit_name = self.kits[index]
        data_list = []
        
        for stem in self.stems:
            ## Search for audio file that starts with sample name
            pattern = os.path.join(self.audio_files_dir, kit_name, f'{stem}*.wav')
            path_list = glob.glob(pattern)
            random_file = random.choice(path_list)

            ## Load audio
            data, sr = load_audio(random_file,sr=self.sr, duration=self.sample_length, approx=True, resample = False)
            data = 0.5 * data[0:1, :] + 0.5 * data[1:, :]
            assert data.shape == (
                self.channels,
                self.sample_length,
            ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
            data_list.append(data)
        
        ## Kit is completed!
        return np.concatenate(data_list, axis=0)

    def get_item(self, index):
        wav = self.get_kit_chunk_randomness(index)
        return self.transform(torch.from_numpy(wav))

    def __len__(self):
        return len(self.kits)

    def __getitem__(self, index):
        return self.get_item(index)