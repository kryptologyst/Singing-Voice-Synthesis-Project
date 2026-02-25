"""
Evaluation script for singing voice synthesis.

This script evaluates trained models and generates samples for analysis.
"""

import os
import argparse
from typing import Dict, Any, List
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tacotron2_hifigan import Tacotron2HiFiGAN
from src.data.singing_dataset import create_dataloader
from src.metrics.evaluation import SVSMetrics
from src.utils import get_device, validate_config


class SVSEvaluator:
    """
    Evaluator for singing voice synthesis models.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
        """
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        self.device = get_device(self.config.device)
        
        # Load model
        self.model = Tacotron2HiFiGAN(self.config.model).to(self.device)
        self._load_checkpoint()
        
        # Initialize metrics
        self.metrics = SVSMetrics(
            sample_rate=self.config.data.audio.sample_rate,
            hop_length=self.config.data.audio.hop_length,
            n_mel_channels=self.config.data.audio.n_mel_channels
        )
        
        # Create output directory
        self.output_dir = Path(self.config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def evaluate(self, split: str = 'test') -> Dict[str, float]:
        """
        Evaluate model on specified split.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dict containing evaluation metrics
        """
        print(f"Evaluating on {split} split...")
        
        # Create data loader
        test_loader = create_dataloader(self.config.data, split)
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute metrics
                batch_metrics = self.metrics.compute(outputs, batch)
                all_metrics.append(batch_metrics)
                
                # Generate samples for first few batches
                if batch_idx < 5:
                    self._generate_samples(batch, outputs, batch_idx)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = np.mean(values)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in avg_metrics.items():
            print(f"{key:20}: {value:.4f}")
        
        # Save results
        results_path = self.output_dir / f'evaluation_{split}.csv'
        pd.DataFrame([avg_metrics]).to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        return avg_metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _generate_samples(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor], 
                         batch_idx: int):
        """Generate sample outputs for analysis."""
        # Save generated audio
        audio_generated = outputs['audio_generated'].cpu().numpy()
        audio_target = batch['audio'].cpu().numpy()
        
        for i in range(min(3, audio_generated.shape[0])):  # Save first 3 samples
            # Generated audio
            generated_path = self.output_dir / f'sample_{batch_idx}_{i}_generated.wav'
            torchaudio.save(
                generated_path,
                torch.from_numpy(audio_generated[i:i+1]),
                self.config.data.audio.sample_rate
            )
            
            # Target audio
            target_path = self.output_dir / f'sample_{batch_idx}_{i}_target.wav'
            torchaudio.save(
                target_path,
                torch.from_numpy(audio_target[i:i+1]),
                self.config.data.audio.sample_rate
            )
        
        # Save mel-spectrograms
        mel_generated = outputs['mel_outputs_postnet'].cpu().numpy()
        mel_target = batch['mel'].cpu().numpy()
        
        for i in range(min(3, mel_generated.shape[0])):
            # Generated mel
            generated_mel_path = self.output_dir / f'sample_{batch_idx}_{i}_mel_generated.npy'
            np.save(generated_mel_path, mel_generated[i])
            
            # Target mel
            target_mel_path = self.output_dir / f'sample_{batch_idx}_{i}_mel_target.npy'
            np.save(target_mel_path, mel_target[i])
    
    def generate_samples_from_text(self, texts: List[str], output_prefix: str = "generated"):
        """
        Generate samples from text inputs.
        
        Args:
            texts: List of input texts
            output_prefix: Prefix for output files
        """
        print(f"Generating samples from {len(texts)} texts...")
        
        from src.data.singing_dataset import TextProcessor
        text_processor = TextProcessor(self.config.data.text_processing)
        
        with torch.no_grad():
            for i, text in enumerate(tqdm(texts, desc="Generating")):
                # Process text
                text_tensor = text_processor.process_text(text)
                text_tensor = text_tensor.unsqueeze(0).to(self.device)
                
                # Generate
                outputs = self.model.inference(text_tensor)
                
                # Save audio
                audio_generated = outputs['audio_generated'].cpu().numpy()
                audio_path = self.output_dir / f'{output_prefix}_{i:03d}.wav'
                torchaudio.save(
                    audio_path,
                    torch.from_numpy(audio_generated),
                    self.config.data.audio.sample_rate
                )
                
                # Save mel-spectrogram
                mel_generated = outputs['mel_outputs_postnet'].cpu().numpy()
                mel_path = self.output_dir / f'{output_prefix}_{i:03d}_mel.npy'
                np.save(mel_path, mel_generated[0])
        
        print(f"Samples saved to: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate singing voice synthesis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--generate_samples', action='store_true', 
                       help='Generate samples from text')
    
    args = parser.parse_args()
    
    # Validate config
    validate_config(OmegaConf.load(args.config))
    
    # Create evaluator
    evaluator = SVSEvaluator(args.config, args.checkpoint)
    
    # Evaluate model
    metrics = evaluator.evaluate(args.split)
    
    # Generate samples if requested
    if args.generate_samples:
        sample_texts = [
            "Hello world, this is a singing voice synthesis demo",
            "The quick brown fox jumps over the lazy dog",
            "Singing voice synthesis is amazing technology",
            "Music and technology come together beautifully",
            "Artificial intelligence creates wonderful melodies"
        ]
        evaluator.generate_samples_from_text(sample_texts)


if __name__ == '__main__':
    main()
