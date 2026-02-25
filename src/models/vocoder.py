"""
HiFi-GAN vocoder for singing voice synthesis.

This module implements the HiFi-GAN neural vocoder for converting mel-spectrograms
to high-quality audio waveforms.
"""

import math
from typing import Tuple, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ResBlock(nn.Module):
    """
    Residual block for HiFi-GAN generator.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int] = (1, 3, 5)):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Kernel size
            dilation: Dilation rates
        """
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, 
                     dilation=d, padding=(kernel_size - 1) // 2 * d)
            for d in dilation
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, 
                     dilation=d, padding=(kernel_size - 1) // 2 * d)
            for d in dilation
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            torch.Tensor: Output tensor
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = self.leaky_relu(x)
            x = conv1(x)
            x = self.leaky_relu(x)
            x = conv2(x)
            x = x + residual
        
        return x


class Generator(nn.Module):
    """
    HiFi-GAN generator.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize generator.
        
        Args:
            config: Generator configuration
        """
        super().__init__()
        
        self.config = config
        vocoder_config = config.vocoder
        
        # Upsampling layers
        self.upsample_rates = [2, 2, 2, 2, 2, 2, 5]
        self.upsample_kernel_sizes = [4, 4, 4, 4, 4, 4, 10]
        self.upsample_initial_channel = vocoder_config.generator_channels
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(
            config.audio.n_mel_channels, 
            self.upsample_initial_channel, 
            7, 1, 3
        )
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    self.upsample_initial_channel // (2 ** i),
                    self.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, (k - u) // 2
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(ResBlock(ch, 3, vocoder_config.generator_dilation_rates[i % len(vocoder_config.generator_dilation_rates)]))
        
        # Final convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input mel-spectrogram [batch, n_mels, time]
            
        Returns:
            torch.Tensor: Generated waveform [batch, 1, samples]
        """
        # Initial convolution
        x = self.conv_pre(x)
        
        # Upsampling and residual blocks
        for up, resblock in zip(self.ups, self.resblocks):
            x = self.leaky_relu(x)
            x = up(x)
            x = resblock(x)
        
        # Final convolution
        x = self.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator for HiFi-GAN.
    """
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        """
        Initialize multi-period discriminator.
        
        Args:
            periods: List of periods for sub-discriminators
        """
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period) for period in periods
        ])
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            y: Real waveform
            y_hat: Generated waveform
            
        Returns:
            Tuple of (real outputs, fake outputs, feature maps)
        """
        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        
        for discriminator in self.discriminators:
            real_out, real_feat = discriminator(y)
            fake_out, fake_feat = discriminator(y_hat)
            
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat)
        
        return real_outputs, fake_outputs, real_feature_maps


class DiscriminatorP(nn.Module):
    """
    Period discriminator.
    """
    
    def __init__(self, period: int):
        """
        Initialize period discriminator.
        
        Args:
            period: Period for this discriminator
        """
        super().__init__()
        
        self.period = period
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), (3, 1), (2, 0)),
        ])
        
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), (1, 1), (1, 0))
        
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input waveform [batch, 1, samples]
            
        Returns:
            Tuple of (output, feature maps)
        """
        # Pad input
        if x.size(2) % self.period != 0:
            pad_length = self.period - (x.size(2) % self.period)
            x = F.pad(x, (0, pad_length))
        
        # Reshape for period-based processing
        batch_size, channels, samples = x.shape
        x = x.view(batch_size, channels, samples // self.period, self.period)
        
        feature_maps = []
        
        # Convolutional layers
        for conv in self.convs:
            x = self.leaky_relu(x)
            feature_maps.append(x)
            x = conv(x)
        
        # Final convolution
        x = self.conv_post(x)
        
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for HiFi-GAN.
    """
    
    def __init__(self):
        """
        Initialize multi-scale discriminator.
        """
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(use_spectral_norm=False),
            DiscriminatorS(use_spectral_norm=False),
        ])
        
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2),
        ])
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            y: Real waveform
            y_hat: Generated waveform
            
        Returns:
            Tuple of (real outputs, fake outputs, feature maps)
        """
        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        
        # Original scale
        real_out, real_feat = self.discriminators[0](y)
        fake_out, fake_feat = self.discriminators[0](y_hat)
        
        real_outputs.append(real_out)
        fake_outputs.append(fake_out)
        real_feature_maps.append(real_feat)
        
        # Downsampled scales
        for i, (discriminator, meanpool) in enumerate(zip(self.discriminators[1:], self.meanpools)):
            y_down = meanpool(y)
            y_hat_down = meanpool(y_hat)
            
            real_out, real_feat = discriminator(y_down)
            fake_out, fake_feat = discriminator(y_hat_down)
            
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat)
        
        return real_outputs, fake_outputs, real_feature_maps


class DiscriminatorS(nn.Module):
    """
    Scale discriminator.
    """
    
    def __init__(self, use_spectral_norm: bool = False):
        """
        Initialize scale discriminator.
        
        Args:
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, 20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, 20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, 20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, 20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, 20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input waveform [batch, 1, samples]
            
        Returns:
            Tuple of (output, feature maps)
        """
        feature_maps = []
        
        # Convolutional layers
        for conv in self.convs:
            x = self.leaky_relu(x)
            feature_maps.append(x)
            x = conv(x)
        
        # Final convolution
        x = self.conv_post(x)
        
        return x, feature_maps


class HiFiGAN(nn.Module):
    """
    HiFi-GAN vocoder.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize HiFi-GAN.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Generator
        self.generator = Generator(config)
        
        # Discriminators
        self.mpd = MultiPeriodDiscriminator(
            periods=config.vocoder.discriminator_periods
        )
        self.msd = MultiScaleDiscriminator()
    
    def forward(self, mel: torch.Tensor, audio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mel: Input mel-spectrogram
            audio: Target audio (for training)
            
        Returns:
            Dict containing model outputs
        """
        # Generate audio
        audio_generated = self.generator(mel)
        
        outputs = {
            'audio_generated': audio_generated
        }
        
        # Discriminator outputs (for training)
        if audio is not None:
            # Multi-period discriminator
            mpd_real, mpd_fake, mpd_real_feat = self.mpd(audio, audio_generated)
            
            # Multi-scale discriminator
            msd_real, msd_fake, msd_real_feat = self.msd(audio, audio_generated)
            
            outputs.update({
                'mpd_real': mpd_real,
                'mpd_fake': mpd_fake,
                'mpd_real_feat': mpd_real_feat,
                'msd_real': msd_real,
                'msd_fake': msd_fake,
                'msd_real_feat': msd_real_feat
            })
        
        return outputs
    
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Inference mode.
        
        Args:
            mel: Input mel-spectrogram
            
        Returns:
            torch.Tensor: Generated audio
        """
        with torch.no_grad():
            return self.generator(mel)
