"""
Dataset and data loading utilities for singing voice synthesis.

This module provides dataset classes and data loading utilities for training
and evaluation of singing voice synthesis models.
"""

import os
import csv
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..features.audio_processor import AudioProcessor, MIDIProcessor
from ..utils import PrivacyGuard


class SingingDataset(Dataset):
    """
    Dataset for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig, split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.wav_dir = self.data_dir / config.wav_dir
        self.midi_dir = self.data_dir / config.midi_dir
        self.meta_file = self.data_dir / config.meta_file
        
        # Initialize processors
        self.audio_processor = AudioProcessor(config.audio)
        self.midi_processor = MIDIProcessor(config.midi_processing)
        self.privacy_guard = PrivacyGuard(
            anonymize_filenames=config.get('anonymize_filenames', True),
            remove_pii=config.get('remove_pii', True)
        )
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        # Text processing
        self.text_processor = TextProcessor(config.text_processing)
        
        # Augmentation settings
        self.augmentation_config = config.get('augmentation', {})
        self.use_augmentation = self.augmentation_config.get('enabled', False) and split == 'train'
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")
        
        metadata = pd.read_csv(self.meta_file)
        
        # Validate required columns
        required_columns = ['id', 'wav_path', 'midi_path', 'lyrics', 'split']
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return metadata
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dict containing sample data
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        wav_path = self.wav_dir / row['wav_path']
        audio = self.audio_processor.load_audio(wav_path)
        
        # Load MIDI
        midi_path = self.midi_dir / row['midi_path']
        midi_data = self.midi_processor.load_midi(midi_path)
        
        # Process text
        text = self.text_processor.process_text(row['lyrics'])
        
        # Extract features
        mel = self.audio_processor.mel_spectrogram(audio)
        f0 = self.audio_processor.extract_f0(audio)
        energy = self.audio_processor.extract_energy(audio)
        
        # Convert MIDI to F0
        midi_f0 = self.midi_processor.notes_to_f0(
            midi_data['notes'],
            midi_data['duration'],
            self.config.audio.hop_length,
            self.config.audio.sample_rate
        )
        
        # Apply augmentation if training
        if self.use_augmentation:
            audio, mel, f0, energy = self._apply_augmentation(audio, mel, f0, energy)
        
        # Create gate signal
        gate = self._create_gate_signal(len(mel[0]))
        
        # Prepare sample
        sample = {
            'id': row['id'],
            'text': text,
            'text_length': torch.tensor(len(text)),
            'mel': mel,
            'mel_length': torch.tensor(mel.shape[1]),
            'audio': audio.squeeze(0),
            'audio_length': torch.tensor(audio.shape[1]),
            'gate': gate,
            'f0': f0,
            'energy': energy,
            'midi_f0': midi_f0,
            'midi_notes': midi_data['notes'],
            'duration': midi_data['duration']
        }
        
        return sample
    
    def _apply_augmentation(self, audio: torch.Tensor, mel: torch.Tensor, 
                           f0: torch.Tensor, energy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Pitch shifting
        if random.random() < 0.5:
            pitch_shift = random.uniform(*self.augmentation_config.get('pitch_shift_range', [-2, 2]))
            audio = self.audio_processor.augment_audio(audio, pitch_shift=pitch_shift)
            mel = self.audio_processor.mel_spectrogram(audio)
            f0 = self.audio_processor.extract_f0(audio)
            energy = self.audio_processor.extract_energy(audio)
        
        # Time stretching
        if random.random() < 0.3:
            time_stretch = random.uniform(*self.augmentation_config.get('time_stretch_range', [0.9, 1.1]))
            audio = self.audio_processor.augment_audio(audio, time_stretch=time_stretch)
            mel = self.audio_processor.mel_spectrogram(audio)
            f0 = self.audio_processor.extract_f0(audio)
            energy = self.audio_processor.extract_energy(audio)
        
        # Noise addition
        if random.random() < 0.2:
            noise_factor = self.augmentation_config.get('noise_factor', 0.01)
            audio = self.audio_processor.augment_audio(audio, noise_factor=noise_factor)
            mel = self.audio_processor.mel_spectrogram(audio)
        
        return audio, mel, f0, energy
    
    def _create_gate_signal(self, mel_length: int) -> torch.Tensor:
        """Create gate signal for mel-spectrogram."""
        gate = torch.zeros(mel_length)
        gate[-1] = 1.0  # Set last frame to 1
        return gate


class TextProcessor:
    """
    Text processing utilities for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize text processor.
        
        Args:
            config: Text processing configuration
        """
        self.config = config
        self.vocab_size = config.phoneme_vocab_size
        self.add_blank = config.add_blank
        self.normalize_text = config.normalize_text
        
        # Create phoneme vocabulary
        self.phoneme_vocab = self._create_phoneme_vocab()
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phoneme_vocab)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phoneme_vocab)}
    
    def _create_phoneme_vocab(self) -> List[str]:
        """Create phoneme vocabulary."""
        # Basic phoneme set for English
        phonemes = [
            'PAD', 'SOS', 'EOS',  # Special tokens
            # Vowels
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
            # Consonants
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
            # Silence
            'SIL', 'SP'
        ]
        
        # Add blank token if needed
        if self.add_blank:
            phonemes.append('BLANK')
        
        return phonemes
    
    def process_text(self, text: str) -> torch.Tensor:
        """
        Process text to phoneme sequence.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Phoneme IDs
        """
        # Normalize text
        if self.normalize_text:
            text = self._normalize_text(text)
        
        # Convert to phonemes (simplified - in practice, use a proper phonemizer)
        phonemes = self._text_to_phonemes(text)
        
        # Convert to IDs
        phoneme_ids = [self.phoneme_to_id.get(p, self.phoneme_to_id['SIL']) for p in phonemes]
        
        # Add special tokens
        phoneme_ids = [self.phoneme_to_id['SOS']] + phoneme_ids + [self.phoneme_to_id['EOS']]
        
        return torch.tensor(phoneme_ids, dtype=torch.long)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text."""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes.
        
        This is a simplified implementation. In practice, you would use
        a proper phonemizer like espeak or CMU Pronouncing Dictionary.
        """
        # Simple mapping for demonstration
        word_to_phonemes = {
            'hello': ['HH', 'EH', 'L', 'OW'],
            'world': ['W', 'ER', 'L', 'D'],
            'singing': ['S', 'IH', 'NG', 'IH', 'NG'],
            'voice': ['V', 'OY', 'S'],
            'synthesis': ['S', 'IH', 'N', 'TH', 'AH', 'S', 'IH', 'S'],
            'the': ['DH', 'AH'],
            'a': ['AH'],
            'and': ['AE', 'N', 'D'],
            'to': ['T', 'UW'],
            'of': ['AH', 'V'],
            'in': ['IH', 'N'],
            'is': ['IH', 'Z'],
            'it': ['IH', 'T'],
            'you': ['Y', 'UW'],
            'that': ['DH', 'AE', 'T'],
            'he': ['HH', 'IY'],
            'was': ['W', 'AH', 'Z'],
            'for': ['F', 'AO', 'R'],
            'on': ['AO', 'N'],
            'are': ['AA', 'R'],
            'as': ['AE', 'Z'],
            'with': ['W', 'IH', 'DH'],
            'his': ['HH', 'IH', 'Z'],
            'they': ['DH', 'EY'],
            'i': ['AY'],
            'at': ['AE', 'T'],
            'be': ['B', 'IY'],
            'this': ['DH', 'IH', 'S'],
            'have': ['HH', 'AE', 'V'],
            'from': ['F', 'R', 'AH', 'M'],
            'or': ['AO', 'R'],
            'one': ['W', 'AH', 'N'],
            'had': ['HH', 'AE', 'D'],
            'by': ['B', 'AY'],
            'word': ['W', 'ER', 'D'],
            'but': ['B', 'AH', 'T'],
            'not': ['N', 'AA', 'T'],
            'what': ['W', 'AH', 'T'],
            'all': ['AO', 'L'],
            'were': ['W', 'ER'],
            'we': ['W', 'IY'],
            'when': ['W', 'EH', 'N'],
            'your': ['Y', 'AO', 'R'],
            'can': ['K', 'AE', 'N'],
            'said': ['S', 'AE', 'D'],
            'there': ['DH', 'EH', 'R'],
            'each': ['IY', 'CH'],
            'which': ['W', 'IH', 'CH'],
            'she': ['SH', 'IY'],
            'do': ['D', 'UW'],
            'how': ['HH', 'AW'],
            'their': ['DH', 'EH', 'R'],
            'if': ['IH', 'F'],
            'will': ['W', 'IH', 'L'],
            'up': ['AH', 'P'],
            'other': ['AH', 'DH', 'ER'],
            'about': ['AH', 'B', 'AW', 'T'],
            'out': ['AW', 'T'],
            'many': ['M', 'EH', 'N', 'IY'],
            'then': ['DH', 'EH', 'N'],
            'them': ['DH', 'EH', 'M'],
            'these': ['DH', 'IY', 'Z'],
            'so': ['S', 'OW'],
            'some': ['S', 'AH', 'M'],
            'her': ['HH', 'ER'],
            'would': ['W', 'UH', 'D'],
            'make': ['M', 'EY', 'K'],
            'like': ['L', 'AY', 'K'],
            'into': ['IH', 'N', 'T', 'UW'],
            'him': ['HH', 'IH', 'M'],
            'time': ['T', 'AY', 'M'],
            'has': ['HH', 'AE', 'Z'],
            'two': ['T', 'UW'],
            'more': ['M', 'AO', 'R'],
            'go': ['G', 'OW'],
            'no': ['N', 'OW'],
            'way': ['W', 'EY'],
            'could': ['K', 'UH', 'D'],
            'my': ['M', 'AY'],
            'than': ['DH', 'AE', 'N'],
            'first': ['F', 'ER', 'S', 'T'],
            'been': ['B', 'IH', 'N'],
            'call': ['K', 'AO', 'L'],
            'who': ['HH', 'UW'],
            'its': ['IH', 'T', 'S'],
            'now': ['N', 'AW'],
            'find': ['F', 'AY', 'N', 'D'],
            'long': ['L', 'AO', 'NG'],
            'down': ['D', 'AW', 'N'],
            'day': ['D', 'EY'],
            'did': ['D', 'IH', 'D'],
            'get': ['G', 'EH', 'T'],
            'come': ['K', 'AH', 'M'],
            'made': ['M', 'EY', 'D'],
            'may': ['M', 'EY'],
            'part': ['P', 'AA', 'R', 'T']
        }
        
        words = text.split()
        phonemes = []
        
        for word in words:
            if word in word_to_phonemes:
                phonemes.extend(word_to_phonemes[word])
            else:
                # Fallback: simple character-to-phoneme mapping
                for char in word:
                    if char.isalpha():
                        phonemes.append(char.upper())
                    else:
                        phonemes.append('SIL')
            phonemes.append('SP')  # Word boundary
        
        return phonemes


def singing_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for singing voice synthesis dataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Dict containing batched tensors
    """
    # Separate different types of data
    texts = [sample['text'] for sample in batch]
    mels = [sample['mel'] for sample in batch]
    audios = [sample['audio'] for sample in batch]
    gates = [sample['gate'] for sample in batch]
    f0s = [sample['f0'] for sample in batch]
    energies = [sample['energy'] for sample in batch]
    midi_f0s = [sample['midi_f0'] for sample in batch]
    
    # Pad sequences
    text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    gate_padded = pad_sequence(gates, batch_first=True, padding_value=0)
    f0_padded = pad_sequence(f0s, batch_first=True, padding_value=0)
    energy_padded = pad_sequence(energies, batch_first=True, padding_value=0)
    midi_f0_padded = pad_sequence(midi_f0s, batch_first=True, padding_value=0)
    
    # Get lengths
    text_lengths = torch.tensor([len(text) for text in texts])
    mel_lengths = torch.tensor([mel.shape[1] for mel in mels])
    audio_lengths = torch.tensor([len(audio) for audio in audios])
    
    return {
        'text': text_padded,
        'text_lengths': text_lengths,
        'mel': mel_padded,
        'mel_lengths': mel_lengths,
        'audio': audio_padded,
        'audio_lengths': audio_lengths,
        'gate': gate_padded,
        'f0': f0_padded,
        'energy': energy_padded,
        'midi_f0': midi_f0_padded,
        'ids': [sample['id'] for sample in batch]
    }


def create_dataloader(config: DictConfig, split: str = 'train') -> DataLoader:
    """
    Create data loader for singing voice synthesis.
    
    Args:
        config: Dataset configuration
        split: Dataset split
        
    Returns:
        DataLoader: Data loader
    """
    dataset = SingingDataset(config, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=singing_collate_fn,
        drop_last=(split == 'train')
    )
    
    return dataloader
