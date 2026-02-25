"""
Streamlit demo application for singing voice synthesis.

This module provides an interactive web interface for demonstrating
the singing voice synthesis system.
"""

import os
import tempfile
import warnings
from typing import Optional, Tuple
from pathlib import Path

import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from omegaconf import OmegaConf

# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tacotron2_hifigan import Tacotron2HiFiGAN
from src.features.audio_processor import AudioProcessor
from src.data.singing_dataset import TextProcessor
from src.utils import get_device, PrivacyGuard


class SVSDemo:
    """
    Singing Voice Synthesis Demo Application.
    """
    
    def __init__(self):
        """Initialize the demo application."""
        self.setup_page_config()
        self.setup_privacy_disclaimer()
        self.load_config()
        self.initialize_model()
        self.setup_processors()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Singing Voice Synthesis Demo",
            page_icon="🎵",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_privacy_disclaimer(self):
        """Setup privacy disclaimer."""
        st.markdown("""
        ## ⚠️ PRIVACY AND ETHICS DISCLAIMER
        
        **This is a research and educational demonstration. This software is NOT intended for:**
        - Production biometric identification
        - Voice cloning or impersonation
        - Commercial applications that could compromise privacy
        
        **By using this demo, you agree to use it responsibly and in accordance with applicable laws.**
        """)
        
        if not st.checkbox("I understand and agree to the privacy disclaimer", value=False):
            st.stop()
    
    def load_config(self):
        """Load model configuration."""
        try:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
            self.config = OmegaConf.load(config_path)
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            st.stop()
    
    def initialize_model(self):
        """Initialize the model."""
        try:
            self.device = get_device(self.config.device)
            self.model = Tacotron2HiFiGAN(self.config.model).to(self.device)
            
            # Load checkpoint if available
            checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                st.success("Model loaded successfully!")
            else:
                st.warning("No trained model found. Using randomly initialized model.")
                
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            st.stop()
    
    def setup_processors(self):
        """Setup audio and text processors."""
        self.audio_processor = AudioProcessor(self.config.data.audio)
        self.text_processor = TextProcessor(self.config.data.text_processing)
        self.privacy_guard = PrivacyGuard()
    
    def run(self):
        """Run the demo application."""
        st.title("🎵 Singing Voice Synthesis Demo")
        
        st.markdown("""
        This demo showcases a neural singing voice synthesis system that converts 
        text lyrics and MIDI melodies into realistic singing voices.
        """)
        
        # Sidebar controls
        self.setup_sidebar()
        
        # Main interface
        self.setup_main_interface()
    
    def setup_sidebar(self):
        """Setup sidebar controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Text input
        st.sidebar.subheader("Lyrics")
        self.lyrics = st.sidebar.text_area(
            "Enter lyrics:",
            value="Hello world, this is a singing voice synthesis demo",
            height=100
        )
        
        # MIDI input options
        st.sidebar.subheader("Melody")
        melody_option = st.sidebar.selectbox(
            "Melody source:",
            ["Simple Scale", "Upload MIDI", "Random Notes"]
        )
        
        if melody_option == "Simple Scale":
            self.melody_notes = self.get_simple_scale()
        elif melody_option == "Upload MIDI":
            self.melody_notes = self.handle_midi_upload()
        else:  # Random Notes
            self.melody_notes = self.generate_random_notes()
        
        # Generation parameters
        st.sidebar.subheader("Generation Parameters")
        self.temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        self.max_length = st.sidebar.slider("Max Length (seconds)", 5, 30, 10)
        
        # Generate button
        if st.sidebar.button("🎵 Generate Singing Voice", type="primary"):
            self.generate_singing_voice()
    
    def get_simple_scale(self) -> list:
        """Get a simple musical scale."""
        # C major scale
        notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        durations = [0.5] * len(notes)
        return list(zip(notes, durations))
    
    def handle_midi_upload(self) -> list:
        """Handle MIDI file upload."""
        uploaded_file = st.sidebar.file_uploader(
            "Upload MIDI file:",
            type=['mid', 'midi'],
            help="Upload a MIDI file to use its melody"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Load MIDI
                import pretty_midi
                midi_data = pretty_midi.PrettyMIDI(tmp_path)
                
                # Extract notes
                notes = []
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            notes.append((note.pitch, note.end - note.start))
                
                # Clean up
                os.unlink(tmp_path)
                
                return notes[:20]  # Limit to first 20 notes
                
            except Exception as e:
                st.sidebar.error(f"Failed to load MIDI: {e}")
                return self.get_simple_scale()
        
        return self.get_simple_scale()
    
    def generate_random_notes(self) -> list:
        """Generate random notes."""
        import random
        
        notes = []
        for _ in range(8):
            pitch = random.randint(60, 84)  # C4 to C6
            duration = random.uniform(0.3, 1.0)
            notes.append((pitch, duration))
        
        return notes
    
    def setup_main_interface(self):
        """Setup main interface."""
        # Display melody visualization
        self.visualize_melody()
        
        # Display generation results
        if hasattr(self, 'generated_audio'):
            self.display_results()
    
    def visualize_melody(self):
        """Visualize the input melody."""
        st.subheader("🎼 Melody Visualization")
        
        if self.melody_notes:
            # Create melody visualization
            fig, ax = plt.subplots(figsize=(12, 4))
            
            pitches = [note[0] for note in self.melody_notes]
            durations = [note[1] for note in self.melody_notes]
            
            # Plot melody
            x_pos = np.cumsum([0] + durations[:-1])
            ax.bar(x_pos, pitches, width=durations, alpha=0.7)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('MIDI Note')
            ax.set_title('Input Melody')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display note information
            st.write("**Note Details:**")
            for i, (pitch, duration) in enumerate(self.melody_notes):
                note_name = self.midi_to_note_name(pitch)
                st.write(f"Note {i+1}: {note_name} (MIDI {pitch}) - {duration:.2f}s")
    
    def midi_to_note_name(self, midi_note: int) -> str:
        """Convert MIDI note number to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = note_names[midi_note % 12]
        return f"{note}{octave}"
    
    def generate_singing_voice(self):
        """Generate singing voice from lyrics and melody."""
        try:
            with st.spinner("Generating singing voice..."):
                # Process text
                text_tensor = self.text_processor.process_text(self.lyrics)
                text_tensor = text_tensor.unsqueeze(0).to(self.device)
                
                # Generate mel-spectrogram
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.inference(text_tensor)
                
                # Generate audio
                audio_generated = outputs['audio_generated']
                
                # Convert to numpy
                audio_np = audio_generated.squeeze().cpu().numpy()
                
                # Store results
                self.generated_audio = audio_np
                self.generated_mel = outputs['mel_outputs_postnet'].squeeze().cpu().numpy()
                self.attention_weights = outputs['alignments'].squeeze().cpu().numpy()
                
                st.success("Singing voice generated successfully!")
                
        except Exception as e:
            st.error(f"Failed to generate singing voice: {e}")
    
    def display_results(self):
        """Display generation results."""
        st.subheader("🎤 Generated Singing Voice")
        
        # Audio player
        st.audio(self.generated_audio, sample_rate=self.config.data.audio.sample_rate)
        
        # Download button
        audio_bytes = self.audio_to_bytes(self.generated_audio)
        st.download_button(
            label="📥 Download Audio",
            data=audio_bytes,
            file_name="generated_singing.wav",
            mime="audio/wav"
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Mel-Spectrogram")
            self.plot_mel_spectrogram()
        
        with col2:
            st.subheader("👁️ Attention Weights")
            self.plot_attention_weights()
        
        # Metrics (if available)
        self.display_metrics()
    
    def plot_mel_spectrogram(self):
        """Plot generated mel-spectrogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mel-spectrogram
        librosa.display.specshow(
            self.generated_mel,
            sr=self.config.data.audio.sample_rate,
            hop_length=self.config.data.audio.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        
        ax.set_title('Generated Mel-Spectrogram')
        plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
        
        st.pyplot(fig)
    
    def plot_attention_weights(self):
        """Plot attention weights."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot attention weights
        im = ax.imshow(
            self.attention_weights,
            aspect='auto',
            origin='lower',
            cmap='Blues'
        )
        
        ax.set_xlabel('Input Text Position')
        ax.set_ylabel('Output Time Position')
        ax.set_title('Attention Weights')
        plt.colorbar(im, ax=ax)
        
        st.pyplot(fig)
    
    def display_metrics(self):
        """Display generation metrics."""
        st.subheader("📊 Generation Metrics")
        
        # Basic metrics
        duration = len(self.generated_audio) / self.config.data.audio.sample_rate
        st.metric("Duration", f"{duration:.2f} seconds")
        
        # Audio statistics
        rms = np.sqrt(np.mean(self.generated_audio ** 2))
        st.metric("RMS Energy", f"{rms:.4f}")
        
        # Mel-spectrogram statistics
        mel_mean = np.mean(self.generated_mel)
        mel_std = np.std(self.generated_mel)
        st.metric("Mel Mean", f"{mel_mean:.4f}")
        st.metric("Mel Std", f"{mel_std:.4f}")
    
    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to bytes for download."""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV file
        import io
        import wave
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.data.audio.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.getvalue()


def main():
    """Main function to run the demo."""
    demo = SVSDemo()
    demo.run()


if __name__ == "__main__":
    main()
