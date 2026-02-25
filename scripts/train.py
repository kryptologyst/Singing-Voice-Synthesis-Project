"""
Training script for singing voice synthesis.

This module provides training utilities and main training loop for the
Tacotron2 + HiFi-GAN singing voice synthesis system.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..models.tacotron2_hifigan import Tacotron2HiFiGAN
from ..data.singing_dataset import create_dataloader
from ..losses.svs_losses import SVSLoss
from ..metrics.evaluation import SVSMetrics
from ..utils import get_device, set_seed, save_config, validate_config, PrivacyGuard


class SVSTrainer:
    """
    Trainer for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Setup device
        self.device = get_device(config.device)
        
        # Setup logging
        self._setup_logging()
        
        # Setup privacy guard
        self.privacy_guard = PrivacyGuard(
            anonymize_filenames=config.privacy.anonymize_filenames,
            remove_pii=config.privacy.remove_pii
        )
        
        # Initialize model
        self.model = Tacotron2HiFiGAN(config.model).to(self.device)
        
        # Initialize loss function
        self.criterion = SVSLoss(config.training.loss_weights)
        
        # Initialize metrics
        self.metrics = SVSMetrics()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader = create_dataloader(config.data, 'train')
        self.val_loader = create_dataloader(config.data, 'val')
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = Path(config.paths.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision
        self.use_amp = config.training.mixed_precision.enabled
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_logging(self) -> None:
        """Setup logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Log privacy disclaimer
        if self.config.privacy.disclaimer_shown:
            self.logger.info("PRIVACY DISCLAIMER: This is a research/educational project. "
                           "Not for production biometric use. See README for details.")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config.training.optimizer
        return OmegaConf.create(optimizer_config)(
            self.model.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_config = self.config.training.scheduler
        return OmegaConf.create(scheduler_config)(
            self.optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict containing training metrics
        """
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'mel_loss': 0.0,
            'gate_loss': 0.0,
            'attention_loss': 0.0,
            'vocoder_mel_loss': 0.0,
            'vocoder_fm_loss': 0.0,
            'vocoder_gen_loss': 0.0,
            'vocoder_disc_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                        loss_dict = self.criterion(outputs, batch)
                else:
                    outputs = self.model(batch)
                    loss_dict = self.criterion(outputs, batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss_dict['total_loss']).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict['total_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                    self.optimizer.step()
                
                # Update metrics
                for key, value in loss_dict.items():
                    epoch_metrics[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'mel': f"{loss_dict['mel_loss'].item():.4f}",
                    'gate': f"{loss_dict['gate_loss'].item():.4f}"
                })
                
                # Log to tensorboard
                if self.global_step % self.config.logging.log_every_n_steps == 0:
                    self._log_training_metrics(loss_dict)
                
                self.global_step += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'mel_loss': 0.0,
            'gate_loss': 0.0,
            'attention_loss': 0.0,
            'vocoder_mel_loss': 0.0,
            'vocoder_fm_loss': 0.0,
            'vocoder_gen_loss': 0.0,
            'vocoder_disc_loss': 0.0,
            'mcd': 0.0,
            'f0_rmse': 0.0,
            'duration_rmse': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch)
                            loss_dict = self.criterion(outputs, batch)
                    else:
                        outputs = self.model(batch)
                        loss_dict = self.criterion(outputs, batch)
                    
                    # Compute metrics
                    metric_dict = self.metrics.compute(outputs, batch)
                    
                    # Update metrics
                    for key, value in loss_dict.items():
                        epoch_metrics[key] += value.item()
                    
                    for key, value in metric_dict.items():
                        epoch_metrics[key] += value
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss'].item():.4f}",
                        'mcd': f"{metric_dict['mcd']:.4f}",
                        'f0_rmse': f"{metric_dict['f0_rmse']:.4f}"
                    })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                device_batch[key] = value
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_training_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        """Log training metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
        
        # Log learning rate
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
    
    def _log_validation_metrics(self, metrics: Dict[str, float]) -> None:
        """Log validation metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {metrics['total_loss']:.4f}")
        
        # Save epoch checkpoint
        if self.current_epoch % self.config.training.checkpointing.save_every_n_epochs == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Set random seed
        set_seed(self.config.seed)
        
        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Save initial config
        save_config(self.config, self.log_dir / 'config.yaml')
        
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Log metrics
            self._log_validation_metrics(val_metrics)
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Check for best model
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"MCD: {val_metrics['mcd']:.4f}, "
                f"F0 RMSE: {val_metrics['f0_rmse']:.4f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.logger.info("Training completed!")
        self.writer.close()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train singing voice synthesis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Validate config
    validate_config(config)
    
    # Create trainer
    trainer = SVSTrainer(config)
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
