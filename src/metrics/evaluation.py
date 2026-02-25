"""
Evaluation metrics for singing voice synthesis.

This module implements various metrics for evaluating the quality of
singing voice synthesis systems.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    warnings.warn("STOI not available. Install with: pip install pystoi")


class MelCepstralDistortion:
    """
    Mel Cepstral Distortion (MCD) metric.
    """
    
    def __init__(self, n_mel_channels: int = 80):
        """
        Initialize MCD calculator.
        
        Args:
            n_mel_channels: Number of mel channels
        """
        self.n_mel_channels = n_mel_channels
    
    def compute(self, mel_pred: torch.Tensor, mel_target: torch.Tensor) -> float:
        """
        Compute MCD between predicted and target mel-spectrograms.
        
        Args:
            mel_pred: Predicted mel-spectrogram [batch, n_mels, time]
            mel_target: Target mel-spectrogram [batch, n_mels, time]
            
        Returns:
            float: MCD value
        """
        # Convert to numpy
        mel_pred_np = mel_pred.detach().cpu().numpy()
        mel_target_np = mel_target.detach().cpu().numpy()
        
        mcd_values = []
        
        for i in range(mel_pred_np.shape[0]):
            pred = mel_pred_np[i]
            target = mel_target_np[i]
            
            # Compute MCD
            diff = pred - target
            mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
            mcd_values.append(mcd)
        
        return np.mean(mcd_values)


class F0RMSE:
    """
    F0 Root Mean Square Error metric.
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 256):
        """
        Initialize F0 RMSE calculator.
        
        Args:
            sample_rate: Sample rate
            hop_length: Hop length
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def compute(self, f0_pred: torch.Tensor, f0_target: torch.Tensor) -> float:
        """
        Compute F0 RMSE between predicted and target F0 contours.
        
        Args:
            f0_pred: Predicted F0 contour [batch, time]
            f0_target: Target F0 contour [batch, time]
            
        Returns:
            float: F0 RMSE value
        """
        # Convert to numpy
        f0_pred_np = f0_pred.detach().cpu().numpy()
        f0_target_np = f0_target.detach().cpu().numpy()
        
        rmse_values = []
        
        for i in range(f0_pred_np.shape[0]):
            pred = f0_pred_np[i]
            target = f0_target_np[i]
            
            # Remove zeros (unvoiced frames)
            mask = (pred > 0) & (target > 0)
            if np.sum(mask) == 0:
                continue
            
            pred_voiced = pred[mask]
            target_voiced = target[mask]
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((pred_voiced - target_voiced) ** 2))
            rmse_values.append(rmse)
        
        return np.mean(rmse_values) if rmse_values else 0.0


class DurationRMSE:
    """
    Duration Root Mean Square Error metric.
    """
    
    def compute(self, duration_pred: torch.Tensor, duration_target: torch.Tensor) -> float:
        """
        Compute duration RMSE between predicted and target durations.
        
        Args:
            duration_pred: Predicted durations [batch, time]
            duration_target: Target durations [batch, time]
            
        Returns:
            float: Duration RMSE value
        """
        # Convert to numpy
        duration_pred_np = duration_pred.detach().cpu().numpy()
        duration_target_np = duration_target.detach().cpu().numpy()
        
        rmse_values = []
        
        for i in range(duration_pred_np.shape[0]):
            pred = duration_pred_np[i]
            target = duration_target_np[i]
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            rmse_values.append(rmse)
        
        return np.mean(rmse_values)


class PESQScore:
    """
    PESQ (Perceptual Evaluation of Speech Quality) metric.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize PESQ calculator.
        
        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
    
    def compute(self, audio_pred: torch.Tensor, audio_target: torch.Tensor) -> float:
        """
        Compute PESQ score between predicted and target audio.
        
        Args:
            audio_pred: Predicted audio [batch, samples]
            audio_target: Target audio [batch, samples]
            
        Returns:
            float: PESQ score
        """
        if not PESQ_AVAILABLE:
            return 0.0
        
        # Convert to numpy
        audio_pred_np = audio_pred.detach().cpu().numpy()
        audio_target_np = audio_target.detach().cpu().numpy()
        
        pesq_values = []
        
        for i in range(audio_pred_np.shape[0]):
            pred = audio_pred_np[i]
            target = audio_target_np[i]
            
            # Ensure same length
            min_len = min(len(pred), len(target))
            pred = pred[:min_len]
            target = target[:min_len]
            
            # Compute PESQ
            try:
                pesq_score = pesq(self.sample_rate, target, pred, 'wb')
                pesq_values.append(pesq_score)
            except Exception:
                continue
        
        return np.mean(pesq_values) if pesq_values else 0.0


class STOIScore:
    """
    STOI (Short-Time Objective Intelligibility) metric.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize STOI calculator.
        
        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
    
    def compute(self, audio_pred: torch.Tensor, audio_target: torch.Tensor) -> float:
        """
        Compute STOI score between predicted and target audio.
        
        Args:
            audio_pred: Predicted audio [batch, samples]
            audio_target: Target audio [batch, samples]
            
        Returns:
            float: STOI score
        """
        if not STOI_AVAILABLE:
            return 0.0
        
        # Convert to numpy
        audio_pred_np = audio_pred.detach().cpu().numpy()
        audio_target_np = audio_target.detach().cpu().numpy()
        
        stoi_values = []
        
        for i in range(audio_pred_np.shape[0]):
            pred = audio_pred_np[i]
            target = audio_target_np[i]
            
            # Ensure same length
            min_len = min(len(pred), len(target))
            pred = pred[:min_len]
            target = target[:min_len]
            
            # Compute STOI
            try:
                stoi_score = stoi(target, pred, self.sample_rate, extended=False)
                stoi_values.append(stoi_score)
            except Exception:
                continue
        
        return np.mean(stoi_values) if stoi_values else 0.0


class SVSMetrics:
    """
    Comprehensive metrics for singing voice synthesis evaluation.
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 256, n_mel_channels: int = 80):
        """
        Initialize SVS metrics.
        
        Args:
            sample_rate: Sample rate
            hop_length: Hop length
            n_mel_channels: Number of mel channels
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        
        # Initialize metric calculators
        self.mcd_calculator = MelCepstralDistortion(n_mel_channels)
        self.f0_rmse_calculator = F0RMSE(sample_rate, hop_length)
        self.duration_rmse_calculator = DurationRMSE()
        self.pesq_calculator = PESQScore(sample_rate)
        self.stoi_calculator = STOIScore(sample_rate)
    
    def compute(self, outputs: Dict[str, torch.Tensor], 
                batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dict containing all computed metrics
        """
        metrics = {}
        
        # Mel-spectrogram metrics
        if 'mel_outputs_postnet' in outputs and 'mel' in batch:
            metrics['mcd'] = self.mcd_calculator.compute(
                outputs['mel_outputs_postnet'], batch['mel']
            )
        
        # F0 metrics
        if 'f0' in batch and 'midi_f0' in batch:
            metrics['f0_rmse'] = self.f0_rmse_calculator.compute(
                batch['f0'], batch['midi_f0']
            )
        
        # Audio quality metrics
        if 'audio_generated' in outputs and 'audio' in batch:
            # Ensure same length for audio comparison
            audio_pred = outputs['audio_generated']
            audio_target = batch['audio']
            
            # Pad or truncate to same length
            max_len = max(audio_pred.shape[1], audio_target.shape[1])
            
            if audio_pred.shape[1] < max_len:
                audio_pred = F.pad(audio_pred, (0, max_len - audio_pred.shape[1]))
            elif audio_pred.shape[1] > max_len:
                audio_pred = audio_pred[:, :max_len]
            
            if audio_target.shape[1] < max_len:
                audio_target = F.pad(audio_target, (0, max_len - audio_target.shape[1]))
            elif audio_target.shape[1] > max_len:
                audio_target = audio_target[:, :max_len]
            
            metrics['pesq'] = self.pesq_calculator.compute(audio_pred, audio_target)
            metrics['stoi'] = self.stoi_calculator.compute(audio_pred, audio_target)
        
        return metrics
    
    def compute_batch_metrics(self, outputs_list: List[Dict[str, torch.Tensor]], 
                             batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Compute metrics for a batch of samples.
        
        Args:
            outputs_list: List of model outputs
            batch_list: List of batch data
            
        Returns:
            Dict containing averaged metrics
        """
        all_metrics = []
        
        for outputs, batch in zip(outputs_list, batch_list):
            metrics = self.compute(outputs, batch)
            all_metrics.append(metrics)
        
        # Average metrics
        averaged_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                averaged_metrics[key] = np.mean(values)
        
        return averaged_metrics
