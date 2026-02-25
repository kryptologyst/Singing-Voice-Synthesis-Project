"""
Combined Tacotron2 + HiFi-GAN model for singing voice synthesis.

This module implements the complete singing voice synthesis system combining
the acoustic model and vocoder.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .acoustic import Tacotron2
from .vocoder import HiFiGAN


class Tacotron2HiFiGAN(nn.Module):
    """
    Combined Tacotron2 + HiFi-GAN model for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the combined model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Acoustic model
        self.acoustic_model = Tacotron2(config)
        
        # Vocoder
        self.vocoder = HiFiGAN(config)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: Batch dictionary containing text, mel, audio, etc.
            
        Returns:
            Dict containing all model outputs
        """
        # Acoustic model forward pass
        acoustic_outputs = self.acoustic_model(batch)
        
        # Vocoder forward pass
        vocoder_outputs = self.vocoder(
            acoustic_outputs['mel_outputs_postnet'],
            batch.get('audio')
        )
        
        # Combine outputs
        outputs = {
            **acoustic_outputs,
            **vocoder_outputs
        }
        
        return outputs
    
    def inference(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference mode.
        
        Args:
            text: Input text tensor
            
        Returns:
            Dict containing generated outputs
        """
        # Acoustic model inference
        acoustic_outputs = self.acoustic_model.inference(text)
        
        # Vocoder inference
        audio_generated = self.vocoder.inference(acoustic_outputs['mel_outputs_postnet'])
        
        return {
            **acoustic_outputs,
            'audio_generated': audio_generated
        }
    
    def get_acoustic_model(self) -> Tacotron2:
        """Get the acoustic model component."""
        return self.acoustic_model
    
    def get_vocoder(self) -> HiFiGAN:
        """Get the vocoder component."""
        return self.vocoder
