"""
Audio processing utilities for singing voice synthesis.

This module provides functions for audio loading, preprocessing, feature extraction,
and augmentation specific to singing voice synthesis tasks.
"""

import os
import warnings
from typing import Tuple, Optional, Union, List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
from scipy import signal
from omegaconf import DictConfig


class AudioProcessor:
    """
    Audio processor for singing voice synthesis.
    
    Handles audio loading, preprocessing, feature extraction, and augmentation.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize audio processor.
        
        Args:
            config: Audio processing configuration
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.hop_length = config.get('hop_length', 256)
        self.win_length = config.get('win_length', 1024)
        self.n_fft = config.get('n_fft', 1024)
        self.n_mel_channels = config.get('n_mel_channels', 80)
        self.mel_fmin = config.get('mel_fmin', 0)
        self.mel_fmax = config.get('mel_fmax', 8000)
        self.min_level_db = config.get('min_level_db', -100)
        self.ref_level_db = config.get('ref_level_db', 20)
        self.power = config.get('power', 1.5)
        
        # Initialize transforms
        self._setup_transforms()
    
    def _setup_transforms(self) -> None:
        """Setup audio transforms."""
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            power=self.power,
            normalized=False
        )
        
        # Inverse mel-spectrogram transform
        self.inv_mel_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mel_channels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax
        )
        
        # Griffin-Lim for waveform reconstruction
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=self.power,
            n_iter=60
        )
    
    def load_audio(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            torch.Tensor: Audio waveform [1, samples]
        """
        try:
            waveform, sr = torchaudio.load(str(path))
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {path}: {e}")
    
    def save_audio(self, waveform: torch.Tensor, path: Union[str, Path]) -> None:
        """
        Save audio waveform to file.
        
        Args:
            waveform: Audio waveform [1, samples] or [samples]
            path: Output file path
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(str(path), waveform, self.sample_rate)
    
    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram from waveform.
        
        Args:
            waveform: Audio waveform [1, samples]
            
        Returns:
            torch.Tensor: Mel-spectrogram [n_mels, time]
        """
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)
    
    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram back to waveform.
        
        Args:
            mel: Mel-spectrogram [n_mels, time]
            
        Returns:
            torch.Tensor: Audio waveform [1, samples]
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        # Convert mel to linear spectrogram
        linear = self.inv_mel_transform(torch.exp(mel))
        
        # Convert to waveform using Griffin-Lim
        waveform = self.griffin_lim(linear)
        
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio waveform.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            torch.Tensor: Normalized waveform
        """
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    
    def trim_silence(self, waveform: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Trim silence from audio waveform.
        
        Args:
            waveform: Audio waveform
            threshold: Silence threshold
            
        Returns:
            torch.Tensor: Trimmed waveform
        """
        # Find non-silent regions
        non_silent = torch.abs(waveform) > threshold
        
        # Find first and last non-silent samples
        indices = torch.nonzero(non_silent)
        if len(indices) == 0:
            return waveform
        
        start_idx = indices[0].item()
        end_idx = indices[-1].item()
        
        return waveform[start_idx:end_idx+1]
    
    def extract_f0(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract fundamental frequency (F0) from waveform.
        
        Args:
            waveform: Audio waveform [samples]
            
        Returns:
            torch.Tensor: F0 contour [time_frames]
        """
        # Convert to numpy for librosa
        audio_np = waveform.squeeze().numpy()
        
        # Extract F0 using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Convert back to torch tensor
        f0_tensor = torch.from_numpy(f0).float()
        
        # Replace NaN values with 0
        f0_tensor = torch.where(torch.isnan(f0_tensor), torch.zeros_like(f0_tensor), f0_tensor)
        
        return f0_tensor
    
    def extract_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract energy contour from waveform.
        
        Args:
            waveform: Audio waveform [samples]
            
        Returns:
            torch.Tensor: Energy contour [time_frames]
        """
        # Compute short-time energy
        energy = torch.stft(
            waveform.squeeze(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        
        # Compute magnitude
        magnitude = torch.abs(energy)
        
        # Sum over frequency bins
        energy_contour = torch.sum(magnitude, dim=0)
        
        return energy_contour
    
    def augment_audio(self, waveform: torch.Tensor, 
                     pitch_shift: Optional[float] = None,
                     time_stretch: Optional[float] = None,
                     noise_factor: Optional[float] = None) -> torch.Tensor:
        """
        Apply audio augmentation.
        
        Args:
            waveform: Audio waveform
            pitch_shift: Pitch shift in semitones
            time_stretch: Time stretch factor
            noise_factor: Noise addition factor
            
        Returns:
            torch.Tensor: Augmented waveform
        """
        augmented = waveform.clone()
        
        # Pitch shifting
        if pitch_shift is not None and pitch_shift != 0:
            augmented = self._pitch_shift(augmented, pitch_shift)
        
        # Time stretching
        if time_stretch is not None and time_stretch != 1.0:
            augmented = self._time_stretch(augmented, time_stretch)
        
        # Noise addition
        if noise_factor is not None and noise_factor > 0:
            augmented = self._add_noise(augmented, noise_factor)
        
        return augmented
    
    def _pitch_shift(self, waveform: torch.Tensor, semitones: float) -> torch.Tensor:
        """Apply pitch shifting."""
        # Convert to numpy for librosa
        audio_np = waveform.squeeze().numpy()
        
        # Apply pitch shifting
        shifted = librosa.effects.pitch_shift(
            audio_np, 
            sr=self.sample_rate, 
            n_steps=semitones
        )
        
        return torch.from_numpy(shifted).unsqueeze(0)
    
    def _time_stretch(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply time stretching."""
        # Convert to numpy for librosa
        audio_np = waveform.squeeze().numpy()
        
        # Apply time stretching
        stretched = librosa.effects.time_stretch(audio_np, rate=factor)
        
        return torch.from_numpy(stretched).unsqueeze(0)
    
    def _add_noise(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(waveform) * factor
        return waveform + noise


class MIDIProcessor:
    """
    MIDI processing utilities for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize MIDI processor.
        
        Args:
            config: MIDI processing configuration
        """
        self.config = config
        self.min_note = config.get('min_note', 21)  # A0
        self.max_note = config.get('max_note', 108)  # C8
        self.note_duration_threshold = config.get('note_duration_threshold', 0.1)
        self.tempo = config.get('tempo', 120)
    
    def load_midi(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load MIDI file and extract note information.
        
        Args:
            path: Path to MIDI file
            
        Returns:
            Dict containing MIDI information
        """
        try:
            import pretty_midi
            
            midi_data = pretty_midi.PrettyMIDI(str(path))
            
            # Extract notes
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append({
                            'pitch': note.pitch,
                            'start': note.start,
                            'end': note.end,
                            'velocity': note.velocity
                        })
            
            # Sort by start time
            notes.sort(key=lambda x: x['start'])
            
            return {
                'notes': notes,
                'tempo': midi_data.estimate_tempo(),
                'duration': midi_data.get_end_time()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MIDI from {path}: {e}")
    
    def notes_to_f0(self, notes: List[Dict], duration: float, 
                   hop_length: int, sample_rate: int) -> torch.Tensor:
        """
        Convert MIDI notes to F0 contour.
        
        Args:
            notes: List of note dictionaries
            duration: Total duration in seconds
            hop_length: Hop length for F0 frames
            sample_rate: Sample rate
            
        Returns:
            torch.Tensor: F0 contour [time_frames]
        """
        n_frames = int(duration * sample_rate / hop_length)
        f0 = torch.zeros(n_frames)
        
        for note in notes:
            start_frame = int(note['start'] * sample_rate / hop_length)
            end_frame = int(note['end'] * sample_rate / hop_length)
            
            # Convert MIDI pitch to frequency
            frequency = 440.0 * (2 ** ((note['pitch'] - 69) / 12))
            
            # Set F0 values for this note
            f0[start_frame:end_frame] = frequency
        
        return f0
    
    def quantize_notes(self, notes: List[Dict], quantization_level: float = 0.25) -> List[Dict]:
        """
        Quantize note timing to regular grid.
        
        Args:
            notes: List of note dictionaries
            quantization_level: Quantization level in seconds
            
        Returns:
            List of quantized note dictionaries
        """
        quantized_notes = []
        
        for note in notes:
            quantized_note = note.copy()
            
            # Quantize start and end times
            quantized_note['start'] = round(note['start'] / quantization_level) * quantization_level
            quantized_note['end'] = round(note['end'] / quantization_level) * quantization_level
            
            # Ensure minimum duration
            if quantized_note['end'] - quantized_note['start'] < self.note_duration_threshold:
                quantized_note['end'] = quantized_note['start'] + self.note_duration_threshold
            
            quantized_notes.append(quantized_note)
        
        return quantized_notes
