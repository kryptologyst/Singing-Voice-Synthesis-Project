"""
Synthetic data generation script for singing voice synthesis.

This script generates a small synthetic dataset for demonstration purposes
when no real dataset is available.
"""

import os
import random
import csv
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features.audio_processor import AudioProcessor, MIDIProcessor
from src.data.singing_dataset import TextProcessor


class SyntheticDataGenerator:
    """
    Generator for synthetic singing voice synthesis data.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize synthetic data generator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = OmegaConf.load(config_path)
        self.data_dir = Path(self.config.data.data_dir)
        
        # Create directories
        self.wav_dir = self.data_dir / self.config.data.wav_dir
        self.midi_dir = self.data_dir / self.config.data.midi_dir
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.midi_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.config.data.audio)
        self.midi_processor = MIDIProcessor(self.config.data.midi_processing)
        self.text_processor = TextProcessor(self.config.data.text_processing)
        
        # Sample lyrics
        self.lyrics_samples = [
            "Hello world, this is a singing voice synthesis demo",
            "The quick brown fox jumps over the lazy dog",
            "Singing voice synthesis is amazing technology",
            "Music and technology come together beautifully",
            "Artificial intelligence creates wonderful melodies",
            "Neural networks learn to sing like humans",
            "Deep learning models generate realistic voices",
            "Machine learning opens new possibilities",
            "Voice synthesis technology is advancing rapidly",
            "Digital music creation is becoming more accessible"
        ]
        
        # Sample melodies (MIDI note sequences)
        self.melody_templates = [
            # Simple ascending scale
            [(60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5), (67, 0.5), (69, 0.5), (71, 0.5), (72, 1.0)],
            # Descending scale
            [(72, 0.5), (71, 0.5), (69, 0.5), (67, 0.5), (65, 0.5), (64, 0.5), (62, 0.5), (60, 1.0)],
            # Arpeggio
            [(60, 0.3), (64, 0.3), (67, 0.3), (72, 0.3), (67, 0.3), (64, 0.3), (60, 1.0)],
            # Random melody
            [(60, 0.4), (64, 0.6), (67, 0.4), (69, 0.8), (65, 0.4), (62, 0.6), (60, 1.0)],
            # Higher register
            [(72, 0.4), (74, 0.4), (76, 0.4), (77, 0.6), (79, 0.4), (81, 0.6), (84, 1.0)]
        ]
    
    def generate_synthetic_audio(self, lyrics: str, melody_notes: List[Tuple[int, float]], 
                               duration: float = 5.0) -> np.ndarray:
        """
        Generate synthetic audio from lyrics and melody.
        
        Args:
            lyrics: Input lyrics
            melody_notes: List of (MIDI_note, duration) tuples
            duration: Total duration in seconds
            
        Returns:
            np.ndarray: Generated audio waveform
        """
        sample_rate = self.config.data.audio.sample_rate
        
        # Generate base melody using sine waves
        melody_audio = self._generate_melody_audio(melody_notes, sample_rate)
        
        # Generate speech-like audio using TTS (simplified)
        speech_audio = self._generate_speech_audio(lyrics, duration, sample_rate)
        
        # Combine melody and speech
        combined_audio = self._combine_audio(melody_audio, speech_audio)
        
        # Add some natural variations
        combined_audio = self._add_natural_variations(combined_audio, sample_rate)
        
        return combined_audio
    
    def _generate_melody_audio(self, melody_notes: List[Tuple[int, float]], 
                             sample_rate: int) -> np.ndarray:
        """Generate melody audio from MIDI notes."""
        audio = np.array([])
        
        for midi_note, duration in melody_notes:
            # Convert MIDI note to frequency
            frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            note_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add envelope
            envelope = np.exp(-t * 2)  # Exponential decay
            note_audio *= envelope
            
            audio = np.concatenate([audio, note_audio])
        
        return audio
    
    def _generate_speech_audio(self, lyrics: str, duration: float, 
                             sample_rate: int) -> np.ndarray:
        """Generate speech-like audio from lyrics."""
        # Simple formant synthesis
        audio = np.zeros(int(sample_rate * duration))
        
        # Create formant frequencies
        f1 = 800  # First formant
        f2 = 1200  # Second formant
        f3 = 2500  # Third formant
        
        # Generate formant-based speech
        t = np.linspace(0, duration, len(audio))
        
        # Base frequency modulation
        f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)  # Varying F0
        
        # Generate harmonics
        for harmonic in range(1, 6):
            harmonic_freq = f0 * harmonic
            harmonic_audio = 0.1 * np.sin(2 * np.pi * harmonic_freq * t)
            
            # Apply formant filtering (simplified)
            if harmonic_freq < f1:
                harmonic_audio *= 0.3
            elif harmonic_freq < f2:
                harmonic_audio *= 0.6
            elif harmonic_freq < f3:
                harmonic_audio *= 0.4
            else:
                harmonic_audio *= 0.2
            
            audio += harmonic_audio
        
        # Add some noise for realism
        noise = 0.01 * np.random.randn(len(audio))
        audio += noise
        
        return audio
    
    def _combine_audio(self, melody_audio: np.ndarray, speech_audio: np.ndarray) -> np.ndarray:
        """Combine melody and speech audio."""
        # Ensure same length
        min_len = min(len(melody_audio), len(speech_audio))
        melody_audio = melody_audio[:min_len]
        speech_audio = speech_audio[:min_len]
        
        # Combine with different weights
        combined = 0.7 * speech_audio + 0.3 * melody_audio
        
        # Normalize
        combined = combined / np.max(np.abs(combined))
        
        return combined
    
    def _add_natural_variations(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add natural variations to audio."""
        # Add slight pitch variations
        t = np.linspace(0, len(audio) / sample_rate, len(audio))
        pitch_variation = 1 + 0.02 * np.sin(2 * np.pi * 0.3 * t)
        
        # Apply pitch variation (simplified)
        audio_varied = audio * pitch_variation
        
        # Add slight amplitude variations
        amplitude_variation = 1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
        audio_varied *= amplitude_variation
        
        # Add reverb (simplified)
        audio_varied = self._add_simple_reverb(audio_varied, sample_rate)
        
        return audio_varied
    
    def _add_simple_reverb(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add simple reverb effect."""
        # Simple delay-based reverb
        delay_samples = int(0.1 * sample_rate)  # 100ms delay
        reverb_gain = 0.3
        
        reverb_audio = audio.copy()
        if len(audio) > delay_samples:
            reverb_audio[delay_samples:] += reverb_gain * audio[:-delay_samples]
        
        return reverb_audio
    
    def create_midi_file(self, melody_notes: List[Tuple[int, float]], 
                        output_path: str) -> None:
        """
        Create MIDI file from melody notes.
        
        Args:
            melody_notes: List of (MIDI_note, duration) tuples
            output_path: Output MIDI file path
        """
        try:
            import pretty_midi
            
            # Create MIDI object
            midi = pretty_midi.PrettyMIDI()
            
            # Create instrument
            instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
            
            # Add notes
            current_time = 0.0
            for midi_note, duration in melody_notes:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                current_time += duration
            
            # Add instrument to MIDI
            midi.instruments.append(instrument)
            
            # Save MIDI file
            midi.write(output_path)
            
        except ImportError:
            print("pretty_midi not available. Skipping MIDI file creation.")
    
    def generate_dataset(self, num_samples: int = 100) -> None:
        """
        Generate synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        metadata = []
        
        for i in tqdm(range(num_samples)):
            # Select random lyrics and melody
            lyrics = random.choice(self.lyrics_samples)
            melody_template = random.choice(self.melody_templates)
            
            # Add some variation to melody
            melody_notes = []
            for note, duration in melody_template:
                # Add slight pitch variation
                pitch_variation = random.randint(-2, 2)
                varied_note = max(21, min(108, note + pitch_variation))
                
                # Add slight duration variation
                duration_variation = random.uniform(0.8, 1.2)
                varied_duration = duration * duration_variation
                
                melody_notes.append((varied_note, varied_duration))
            
            # Generate audio
            duration = sum(duration for _, duration in melody_notes)
            audio = self.generate_synthetic_audio(lyrics, melody_notes, duration)
            
            # Save audio file
            audio_filename = f"sample_{i:04d}.wav"
            audio_path = self.wav_dir / audio_filename
            sf.write(audio_path, audio, self.config.data.audio.sample_rate)
            
            # Save MIDI file
            midi_filename = f"sample_{i:04d}.mid"
            midi_path = self.midi_dir / midi_filename
            self.create_midi_file(melody_notes, str(midi_path))
            
            # Determine split
            if i < int(0.8 * num_samples):
                split = 'train'
            elif i < int(0.9 * num_samples):
                split = 'val'
            else:
                split = 'test'
            
            # Add to metadata
            metadata.append({
                'id': f"sample_{i:04d}",
                'wav_path': audio_filename,
                'midi_path': midi_filename,
                'lyrics': lyrics,
                'singer_id': 'synthetic',
                'split': split,
                'duration': duration,
                'sample_rate': self.config.data.audio.sample_rate
            })
        
        # Save metadata
        metadata_path = self.data_dir / self.config.data.meta_file
        with open(metadata_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'wav_path', 'midi_path', 'lyrics', 'singer_id', 'split', 'duration', 'sample_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
        
        print(f"Dataset generated successfully!")
        print(f"Audio files: {self.wav_dir}")
        print(f"MIDI files: {self.midi_dir}")
        print(f"Metadata: {metadata_path}")
        print(f"Train samples: {len([m for m in metadata if m['split'] == 'train'])}")
        print(f"Val samples: {len([m for m in metadata if m['split'] == 'val'])}")
        print(f"Test samples: {len([m for m in metadata if m['split'] == 'test'])}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic singing voice dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = SyntheticDataGenerator(args.config)
    generator.generate_dataset(args.num_samples)


if __name__ == '__main__':
    main()
