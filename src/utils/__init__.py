"""
Core utilities for the singing voice synthesis project.

This module provides device management, seeding, and other utility functions
for reproducible and cross-platform training.
"""

import os
import random
from typing import Optional, Union, Any, Dict
import warnings

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        device: Device preference ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_audio_backend() -> None:
    """
    Setup audio backend for optimal performance.
    """
    # Set audio backend based on available options
    if hasattr(torchaudio, "set_audio_backend"):
        try:
            torchaudio.set_audio_backend("sox_io")
        except Exception:
            try:
                torchaudio.set_audio_backend("soundfile")
            except Exception:
                warnings.warn("Could not set preferred audio backend")


def get_model_summary(model: torch.nn.Module, input_size: tuple) -> str:
    """
    Get a summary of model parameters and memory usage.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        
    Returns:
        str: Model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Non-trainable parameters: {total_params - trainable_params:,}
- Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)
"""
    return summary


def save_config(config: DictConfig, path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: OmegaConf configuration
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(config.pretty())


def load_config(path: str) -> DictConfig:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path
        
    Returns:
        DictConfig: Loaded configuration
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def anonymize_filename(filename: str) -> str:
    """
    Anonymize filename by removing potentially identifying information.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Anonymized filename
    """
    import hashlib
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Create hash of original name
    hash_obj = hashlib.md5(name.encode())
    anonymized_name = hash_obj.hexdigest()[:8]
    
    return f"{anonymized_name}{ext}"


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate audio parameters
    audio_config = config.data.get('audio', {})
    if audio_config.get('sample_rate', 0) <= 0:
        raise ValueError("Sample rate must be positive")
    
    if audio_config.get('hop_length', 0) <= 0:
        raise ValueError("Hop length must be positive")
    
    # Validate model parameters
    model_config = config.model
    if model_config.get('acoustic_model', {}).get('encoder_dim', 0) <= 0:
        raise ValueError("Encoder dimension must be positive")


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information for experiment tracking.
    
    Returns:
        Dict containing git information
    """
    try:
        import subprocess
        
        result = {}
        result['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        result['branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        result['dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip())
        
        return result
    except Exception:
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}


class PrivacyGuard:
    """
    Privacy guard for handling sensitive data in logs and outputs.
    """
    
    def __init__(self, anonymize_filenames: bool = True, remove_pii: bool = True):
        self.anonymize_filenames = anonymize_filenames
        self.remove_pii = remove_pii
    
    def sanitize_log_message(self, message: str) -> str:
        """
        Remove potentially identifying information from log messages.
        
        Args:
            message: Original log message
            
        Returns:
            str: Sanitized log message
        """
        if not self.remove_pii:
            return message
        
        # Remove common PII patterns
        import re
        
        # Remove email addresses
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                        '[EMAIL]', message)
        
        # Remove phone numbers
        message = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', message)
        
        # Remove SSN patterns
        message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', message)
        
        return message
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for logging.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        if self.anonymize_filenames:
            return anonymize_filename(filename)
        return filename
