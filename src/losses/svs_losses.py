"""
Loss functions for singing voice synthesis.

This module implements various loss functions used in training the
Tacotron2 + HiFi-GAN singing voice synthesis system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from omegaconf import DictConfig


class MelLoss(nn.Module):
    """
    Mel-spectrogram reconstruction loss.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mel_pred: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram loss.
        
        Args:
            mel_pred: Predicted mel-spectrogram [batch, n_mels, time]
            mel_target: Target mel-spectrogram [batch, n_mels, time]
            
        Returns:
            torch.Tensor: Mel loss
        """
        return F.l1_loss(mel_pred, mel_target)


class GateLoss(nn.Module):
    """
    Gate loss for stopping token prediction.
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, gate_pred: torch.Tensor, gate_target: torch.Tensor) -> torch.Tensor:
        """
        Compute gate loss.
        
        Args:
            gate_pred: Predicted gate logits [batch, time]
            gate_target: Target gate values [batch, time]
            
        Returns:
            torch.Tensor: Gate loss
        """
        return self.bce_loss(gate_pred, gate_target)


class AttentionLoss(nn.Module):
    """
    Attention loss for encouraging diagonal attention patterns.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, alignments: torch.Tensor) -> torch.Tensor:
        """
        Compute attention loss.
        
        Args:
            alignments: Attention alignments [batch, time_out, time_in]
            
        Returns:
            torch.Tensor: Attention loss
        """
        # Encourage diagonal attention patterns
        batch_size, time_out, time_in = alignments.shape
        
        # Create diagonal mask
        diagonal_mask = torch.zeros_like(alignments)
        for i in range(time_out):
            j = min(i * time_in // time_out, time_in - 1)
            diagonal_mask[:, i, j] = 1.0
        
        # Compute loss
        loss = F.mse_loss(alignments, diagonal_mask)
        
        return loss


class GeneratorLoss(nn.Module):
    """
    Generator loss for HiFi-GAN.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, fake_outputs: list, real_feature_maps: list) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_outputs: List of fake discriminator outputs
            real_feature_maps: List of real feature maps
            
        Returns:
            torch.Tensor: Generator loss
        """
        loss = 0.0
        
        for fake_out in fake_outputs:
            # Adversarial loss
            loss += F.mse_loss(fake_out, torch.ones_like(fake_out))
        
        return loss


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for HiFi-GAN.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, real_outputs: list, fake_outputs: list) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_outputs: List of real discriminator outputs
            fake_outputs: List of fake discriminator outputs
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            # Real loss
            loss += F.mse_loss(real_out, torch.ones_like(real_out))
            
            # Fake loss
            loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for HiFi-GAN generator.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, fake_feature_maps: list, real_feature_maps: list) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            fake_feature_maps: List of fake feature maps
            real_feature_maps: List of real feature maps
            
        Returns:
            torch.Tensor: Feature matching loss
        """
        loss = 0.0
        
        for fake_feat, real_feat in zip(fake_feature_maps, real_feature_maps):
            for fake_layer, real_layer in zip(fake_feat, real_feat):
                loss += F.l1_loss(fake_layer, real_layer)
        
        return loss


class SVSLoss(nn.Module):
    """
    Combined loss function for singing voice synthesis.
    """
    
    def __init__(self, loss_weights: DictConfig):
        """
        Initialize SVS loss.
        
        Args:
            loss_weights: Loss weight configuration
        """
        super().__init__()
        
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        self.mel_loss = MelLoss()
        self.gate_loss = GateLoss()
        self.attention_loss = AttentionLoss()
        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        
        # Acoustic model losses
        losses['mel_loss'] = self.mel_loss(
            outputs['mel_outputs'], batch['mel']
        )
        
        losses['mel_loss_postnet'] = self.mel_loss(
            outputs['mel_outputs_postnet'], batch['mel']
        )
        
        losses['gate_loss'] = self.gate_loss(
            outputs['gate_outputs'], batch['gate']
        )
        
        losses['attention_loss'] = self.attention_loss(
            outputs['alignments']
        )
        
        # Vocoder losses
        if 'mpd_real' in outputs and 'mpd_fake' in outputs:
            losses['vocoder_gen_loss'] = self.generator_loss(
                outputs['mpd_fake'], outputs['mpd_real_feat']
            ) + self.generator_loss(
                outputs['msd_fake'], outputs['msd_real_feat']
            )
            
            losses['vocoder_disc_loss'] = self.discriminator_loss(
                outputs['mpd_real'], outputs['mpd_fake']
            ) + self.discriminator_loss(
                outputs['msd_real'], outputs['msd_fake']
            )
            
            losses['vocoder_fm_loss'] = self.feature_matching_loss(
                outputs['mpd_fake'], outputs['mpd_real_feat']
            ) + self.feature_matching_loss(
                outputs['msd_fake'], outputs['msd_real_feat']
            )
        
        # Compute total loss
        total_loss = (
            self.loss_weights.mel_loss * losses['mel_loss'] +
            self.loss_weights.mel_loss * losses['mel_loss_postnet'] +
            self.loss_weights.gate_loss * losses['gate_loss'] +
            self.loss_weights.attention_loss * losses['attention_loss']
        )
        
        if 'vocoder_gen_loss' in losses:
            total_loss += (
                self.loss_weights.vocoder_gen_loss * losses['vocoder_gen_loss'] +
                self.loss_weights.vocoder_disc_loss * losses['vocoder_disc_loss'] +
                self.loss_weights.vocoder_fm_loss * losses['vocoder_fm_loss']
            )
        
        losses['total_loss'] = total_loss
        
        return losses
