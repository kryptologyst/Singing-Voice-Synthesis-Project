# Singing Voice Synthesis Project

## PRIVACY AND ETHICS DISCLAIMER

**IMPORTANT: This is a research and educational demonstration project. This software is NOT intended for production use in biometric identification, voice cloning, or any commercial applications that could compromise privacy or security.**

### Prohibited Uses:
- Creating deepfakes or impersonating real individuals
- Biometric identification or verification systems
- Commercial voice cloning services
- Any application that could be used to deceive or harm individuals

### Intended Use:
- Academic research in singing voice synthesis
- Educational demonstrations of neural audio generation
- Non-commercial experimentation with voice synthesis techniques

By using this software, you agree to use it responsibly and in accordance with applicable laws and ethical guidelines.

## Overview

This project implements a singing voice synthesis (SVS) system using neural networks. The system converts MIDI sequences and lyrics into realistic singing voices using:

- **Acoustic Model**: Tacotron2-style encoder-decoder architecture
- **Vocoder**: HiFi-GAN neural vocoder for high-quality audio generation
- **Features**: Mel-spectrograms, F0 contours, and linguistic features

## Quick Start

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Dataset** (if no dataset available):
   ```bash
   python scripts/generate_synthetic_data.py
   ```

3. **Train the Model**:
   ```bash
   python scripts/train.py --config configs/tacotron2_hifigan.yaml
   ```

4. **Run Demo**:
   ```bash
   streamlit run demo/app.py
   ```

## Dataset Schema

The project expects data in the following structure:
```
data/
├── wav/           # Audio files (16kHz, mono)
├── midi/          # MIDI files with melody and timing
├── meta.csv       # Metadata with columns: id, wav_path, midi_path, lyrics, singer_id, split
└── annotations/   # Optional: phoneme alignments, F0 annotations
```

## Training

```bash
# Basic training
python scripts/train.py --config configs/tacotron2_hifigan.yaml

# Resume from checkpoint
python scripts/train.py --config configs/tacotron2_hifigan.yaml --resume checkpoints/latest.pth

# Multi-GPU training
python scripts/train.py --config configs/tacotron2_hifigan.yaml --gpus 2
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --config configs/tacotron2_hifigan.yaml --checkpoint checkpoints/best.pth

# Generate samples
python scripts/generate_samples.py --config configs/tacotron2_hifigan.yaml --checkpoint checkpoints/best.pth
```

## Metrics

The system evaluates singing voice synthesis quality using:

- **MCD (Mel Cepstral Distortion)**: Spectral quality
- **F0 RMSE**: Pitch accuracy
- **FAD (Fréchet Audio Distance)**: Perceptual quality
- **Duration RMSE**: Timing accuracy

## Model Architecture

### Acoustic Model (Tacotron2-style)
- Encoder: Convolutional + BiLSTM
- Attention: Location-aware attention mechanism
- Decoder: Pre-net + LSTM + Linear projection
- Post-net: 5-layer CNN for mel-spectrogram refinement

### Vocoder (HiFi-GAN)
- Generator: Multi-receptive field fusion (MRF)
- Discriminator: Multi-period discriminator (MPD) + Multi-scale discriminator (MSD)
- Loss: Adversarial + Feature matching + Mel-spectrogram

## Configuration

Key configuration parameters in `configs/tacotron2_hifigan.yaml`:

```yaml
data:
  sample_rate: 22050
  hop_length: 256
  win_length: 1024
  n_mel_channels: 80
  
model:
  acoustic:
    encoder_dim: 512
    decoder_dim: 1024
    attention_dim: 128
  vocoder:
    generator_channels: 512
    discriminator_channels: 64
```

## Limitations

- Requires substantial training data for high-quality results
- Performance depends on MIDI quality and phoneme alignment
- May not generalize well to unseen singers or musical styles
- Computational requirements: GPU recommended for training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{singing_voice_synthesis,
  title={Singing Voice Synthesis with Neural Networks},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Singing-Voice-Synthesis-Project}
}
```
# Singing-Voice-Synthesis-Project
