"""
Tests for singing voice synthesis package.

This module contains unit tests for the various components of the
singing voice synthesis system.
"""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tacotron2_hifigan import Tacotron2HiFiGAN
from src.models.acoustic import Tacotron2, Encoder, Decoder, Postnet
from src.models.vocoder import HiFiGAN, Generator, MultiPeriodDiscriminator
from src.losses.svs_losses import SVSLoss, MelLoss, GateLoss, AttentionLoss
from src.metrics.evaluation import SVSMetrics, MelCepstralDistortion, F0RMSE
from src.features.audio_processor import AudioProcessor
from src.data.singing_dataset import TextProcessor
from src.utils import get_device, set_seed


class TestModels:
    """Test model components."""
    
    def test_encoder(self):
        """Test encoder component."""
        vocab_size = 100
        encoder_dim = 512
        
        encoder = Encoder(vocab_size, encoder_dim)
        
        # Test forward pass
        batch_size = 2
        seq_len = 50
        text = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.tensor([seq_len, seq_len])
        
        outputs, lengths = encoder(text, input_lengths)
        
        assert outputs.shape == (batch_size, seq_len, encoder_dim)
        assert lengths.shape == (batch_size,)
    
    def test_decoder(self):
        """Test decoder component."""
        n_mel_channels = 80
        encoder_dim = 512
        decoder_dim = 1024
        
        decoder = Decoder(n_mel_channels, encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        
        # Test forward pass
        batch_size = 2
        seq_len = 50
        mel_len = 100
        
        memory = torch.randn(batch_size, seq_len, encoder_dim)
        decoder_inputs = torch.randn(batch_size, mel_len, n_mel_channels)
        memory_lengths = torch.tensor([seq_len, seq_len])
        
        mel_outputs, gate_outputs, alignments = decoder(memory, decoder_inputs, memory_lengths)
        
        assert mel_outputs.shape == (batch_size, mel_len, n_mel_channels)
        assert gate_outputs.shape == (batch_size, mel_len, 1)
        assert alignments.shape == (batch_size, mel_len, seq_len)
    
    def test_postnet(self):
        """Test postnet component."""
        n_mel_channels = 80
        postnet_dim = 512
        
        postnet = Postnet(n_mel_channels, postnet_dim)
        
        # Test forward pass
        batch_size = 2
        mel_len = 100
        
        mel_input = torch.randn(batch_size, n_mel_channels, mel_len)
        mel_output = postnet(mel_input)
        
        assert mel_output.shape == mel_input.shape
    
    def test_generator(self):
        """Test HiFi-GAN generator."""
        config = OmegaConf.create({
            'audio': {'n_mel_channels': 80},
            'vocoder': {
                'generator_channels': 512,
                'generator_dilation_rates': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            }
        })
        
        generator = Generator(config)
        
        # Test forward pass
        batch_size = 2
        mel_len = 100
        
        mel_input = torch.randn(batch_size, 80, mel_len)
        audio_output = generator(mel_input)
        
        assert audio_output.shape[0] == batch_size
        assert audio_output.shape[1] == 1  # Mono audio
        assert audio_output.shape[2] > mel_len  # Upsampled
    
    def test_tacotron2(self):
        """Test Tacotron2 model."""
        config = OmegaConf.create({
            'acoustic_model': {
                'encoder_dim': 512,
                'encoder_n_convs': 3,
                'encoder_conv_dim': 512,
                'attention_rnn_dim': 1024,
                'attention_dim': 128,
                'attention_location_n_filters': 32,
                'attention_location_kernel_size': 31,
                'decoder_rnn_dim': 1024,
                'prenet_dim': 256,
                'max_decoder_steps': 1000,
                'gate_threshold': 0.5,
                'postnet_conv_dim': 512,
                'postnet_kernel_size': 5,
                'postnet_n_convs': 5
            },
            'audio': {'n_mel_channels': 80},
            'text_processing': {'phoneme_vocab_size': 100}
        })
        
        model = Tacotron2(config)
        
        # Test forward pass
        batch = {
            'text': torch.randint(0, 100, (2, 50)),
            'text_lengths': torch.tensor([50, 50]),
            'mel': torch.randn(2, 80, 100),
            'gate': torch.randn(2, 100)
        }
        
        outputs = model(batch)
        
        assert 'mel_outputs' in outputs
        assert 'mel_outputs_postnet' in outputs
        assert 'gate_outputs' in outputs
        assert 'alignments' in outputs


class TestLosses:
    """Test loss functions."""
    
    def test_mel_loss(self):
        """Test mel-spectrogram loss."""
        mel_loss = MelLoss()
        
        batch_size = 2
        n_mels = 80
        time_len = 100
        
        mel_pred = torch.randn(batch_size, n_mels, time_len)
        mel_target = torch.randn(batch_size, n_mels, time_len)
        
        loss = mel_loss(mel_pred, mel_target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_gate_loss(self):
        """Test gate loss."""
        gate_loss = GateLoss()
        
        batch_size = 2
        time_len = 100
        
        gate_pred = torch.randn(batch_size, time_len)
        gate_target = torch.randn(batch_size, time_len)
        
        loss = gate_loss(gate_pred, gate_target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_attention_loss(self):
        """Test attention loss."""
        attention_loss = AttentionLoss()
        
        batch_size = 2
        time_out = 100
        time_in = 50
        
        alignments = torch.randn(batch_size, time_out, time_in)
        alignments = torch.softmax(alignments, dim=-1)  # Normalize
        
        loss = attention_loss(alignments)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_svs_loss(self):
        """Test combined SVS loss."""
        loss_weights = OmegaConf.create({
            'mel_loss': 1.0,
            'gate_loss': 1.0,
            'attention_loss': 0.1,
            'vocoder_mel_loss': 45.0,
            'vocoder_fm_loss': 2.0,
            'vocoder_gen_loss': 1.0,
            'vocoder_disc_loss': 1.0
        })
        
        svs_loss = SVSLoss(loss_weights)
        
        # Mock outputs and batch
        outputs = {
            'mel_outputs': torch.randn(2, 80, 100),
            'mel_outputs_postnet': torch.randn(2, 80, 100),
            'gate_outputs': torch.randn(2, 100),
            'alignments': torch.randn(2, 100, 50)
        }
        
        batch = {
            'mel': torch.randn(2, 80, 100),
            'gate': torch.randn(2, 100)
        }
        
        losses = svs_loss(outputs, batch)
        
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].item() >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_mcd(self):
        """Test Mel Cepstral Distortion metric."""
        mcd = MelCepstralDistortion(n_mel_channels=80)
        
        batch_size = 2
        n_mels = 80
        time_len = 100
        
        mel_pred = torch.randn(batch_size, n_mels, time_len)
        mel_target = torch.randn(batch_size, n_mels, time_len)
        
        mcd_value = mcd.compute(mel_pred, mel_target)
        
        assert isinstance(mcd_value, float)
        assert mcd_value >= 0
    
    def test_f0_rmse(self):
        """Test F0 RMSE metric."""
        f0_rmse = F0RMSE(sample_rate=22050, hop_length=256)
        
        batch_size = 2
        time_len = 100
        
        f0_pred = torch.randn(batch_size, time_len)
        f0_target = torch.randn(batch_size, time_len)
        
        rmse_value = f0_rmse.compute(f0_pred, f0_target)
        
        assert isinstance(rmse_value, float)
        assert rmse_value >= 0
    
    def test_svs_metrics(self):
        """Test SVS metrics."""
        metrics = SVSMetrics(sample_rate=22050, hop_length=256, n_mel_channels=80)
        
        outputs = {
            'mel_outputs_postnet': torch.randn(2, 80, 100),
            'audio_generated': torch.randn(2, 1000)
        }
        
        batch = {
            'mel': torch.randn(2, 80, 100),
            'audio': torch.randn(2, 1000),
            'f0': torch.randn(2, 100),
            'midi_f0': torch.randn(2, 100)
        }
        
        metric_dict = metrics.compute(outputs, batch)
        
        assert isinstance(metric_dict, dict)
        assert 'mcd' in metric_dict
        assert 'f0_rmse' in metric_dict


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    def test_audio_processor(self):
        """Test audio processor."""
        config = OmegaConf.create({
            'sample_rate': 22050,
            'hop_length': 256,
            'win_length': 1024,
            'n_fft': 1024,
            'n_mel_channels': 80,
            'mel_fmin': 0,
            'mel_fmax': 8000,
            'min_level_db': -100,
            'ref_level_db': 20,
            'power': 1.5
        })
        
        processor = AudioProcessor(config)
        
        # Test mel-spectrogram computation
        batch_size = 2
        samples = 10000
        
        waveform = torch.randn(batch_size, samples)
        mel = processor.mel_spectrogram(waveform)
        
        assert mel.shape[0] == batch_size
        assert mel.shape[1] == config.n_mel_channels
    
    def test_text_processor(self):
        """Test text processor."""
        config = OmegaConf.create({
            'phoneme_vocab_size': 100,
            'add_blank': True,
            'normalize_text': True
        })
        
        processor = TextProcessor(config)
        
        # Test text processing
        text = "Hello world"
        phoneme_ids = processor.process_text(text)
        
        assert isinstance(phoneme_ids, torch.Tensor)
        assert phoneme_ids.dtype == torch.long
        assert len(phoneme_ids) > 0


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


if __name__ == '__main__':
    pytest.main([__file__])
