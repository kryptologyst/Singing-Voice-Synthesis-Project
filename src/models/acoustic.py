"""
Tacotron2-style acoustic model for singing voice synthesis.

This module implements the acoustic model component of the singing voice synthesis
system, based on the Tacotron2 architecture but adapted for singing voice.
"""

import math
from typing import Tuple, Optional, Dict, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig


class Encoder(nn.Module):
    """
    Tacotron2 encoder for text/phoneme input.
    """
    
    def __init__(self, vocab_size: int, encoder_dim: int = 512, 
                 encoder_n_convs: int = 3, encoder_conv_dim: int = 512):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Size of input vocabulary
            encoder_dim: Encoder embedding dimension
            encoder_n_convs: Number of convolutional layers
            encoder_conv_dim: Convolutional layer dimension
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        
        # Convolutional layers
        conv_layers = []
        for i in range(encoder_n_convs):
            conv_layers.append(
                nn.Conv1d(
                    encoder_dim,
                    encoder_dim,
                    kernel_size=5,
                    padding=2
                )
            )
            conv_layers.append(nn.BatchNorm1d(encoder_dim))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(0.5))
        
        self.convolutions = nn.Sequential(*conv_layers)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            encoder_dim,
            encoder_dim // 2,
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input token sequences [batch, seq_len]
            input_lengths: Input sequence lengths [batch]
            
        Returns:
            Tuple of (encoded features, encoded lengths)
        """
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, encoder_dim]
        
        # Transpose for convolution
        x = x.transpose(1, 2)  # [batch, encoder_dim, seq_len]
        
        # Convolutional layers
        x = self.convolutions(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)  # [batch, seq_len, encoder_dim]
        
        # Pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, input_lengths


class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention mechanism.
    """
    
    def __init__(self, attention_rnn_dim: int = 1024, encoder_dim: int = 512,
                 attention_dim: int = 128, attention_location_n_filters: int = 32,
                 attention_location_kernel_size: int = 31):
        """
        Initialize attention mechanism.
        
        Args:
            attention_rnn_dim: Attention RNN dimension
            encoder_dim: Encoder dimension
            attention_dim: Attention dimension
            attention_location_n_filters: Number of location filters
            attention_location_kernel_size: Location kernel size
        """
        super().__init__()
        
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        # Location-sensitive attention
        self.location_conv = nn.Conv1d(
            2, attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=(attention_location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_layer = nn.Linear(
            attention_location_n_filters, attention_dim, bias=False
        )
        
        self.score_mask_value = -float("inf")
    
    def get_alignment_energies(self, query: torch.Tensor, processed_memory: torch.Tensor,
                             attention_weights_cat: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment energies.
        
        Args:
            query: Query tensor [batch, attention_dim]
            processed_memory: Processed memory [batch, seq_len, attention_dim]
            attention_weights_cat: Previous attention weights [batch, 2, seq_len]
            
        Returns:
            torch.Tensor: Alignment energies [batch, seq_len]
        """
        # Location features
        processed_attention_weights = self.location_conv(attention_weights_cat)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_layer(processed_attention_weights)
        
        # Energy computation
        energies = self.v(torch.tanh(
            query.unsqueeze(1) + processed_memory + processed_attention_weights
        ))
        
        return energies.squeeze(-1)
    
    def forward(self, attention_hidden_state: torch.Tensor, memory: torch.Tensor,
                processed_memory: torch.Tensor, attention_weights_cat: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            attention_hidden_state: Hidden state from attention RNN
            memory: Encoder memory
            processed_memory: Processed encoder memory
            attention_weights_cat: Previous attention weights
            mask: Attention mask
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # Process query
        query = self.query_layer(attention_hidden_state)
        
        # Compute alignment energies
        alignment = self.get_alignment_energies(
            query, processed_memory, attention_weights_cat
        )
        
        # Apply mask
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        
        # Compute attention weights
        attention_weights = F.softmax(alignment, dim=1)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), memory)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights


class Decoder(nn.Module):
    """
    Tacotron2 decoder for mel-spectrogram generation.
    """
    
    def __init__(self, n_mel_channels: int = 80, n_frames_per_step: int = 1,
                 encoder_dim: int = 512, decoder_dim: int = 1024,
                 prenet_dim: int = 256, attention_rnn_dim: int = 1024,
                 attention_dim: int = 128, attention_location_n_filters: int = 32,
                 attention_location_kernel_size: int = 31, max_decoder_steps: int = 1000,
                 gate_threshold: float = 0.5):
        """
        Initialize decoder.
        
        Args:
            n_mel_channels: Number of mel channels
            n_frames_per_step: Number of frames per decoder step
            encoder_dim: Encoder dimension
            decoder_dim: Decoder dimension
            prenet_dim: Prenet dimension
            attention_rnn_dim: Attention RNN dimension
            attention_dim: Attention dimension
            attention_location_n_filters: Number of location filters
            attention_location_kernel_size: Location kernel size
            max_decoder_steps: Maximum decoder steps
            gate_threshold: Gate threshold
        """
        super().__init__()
        
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        
        # Prenet
        self.prenet = nn.Sequential(
            nn.Linear(n_mel_channels, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_dim, attention_rnn_dim
        )
        
        # Attention mechanism
        self.attention_layer = LocationSensitiveAttention(
            attention_rnn_dim, encoder_dim, attention_dim,
            attention_location_n_filters, attention_location_kernel_size
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_dim, decoder_dim, bias=True
        )
        
        # Linear projections
        self.linear_projection = nn.Linear(
            decoder_dim + encoder_dim, n_mel_channels * n_frames_per_step
        )
        
        self.gate_layer = nn.Linear(decoder_dim + encoder_dim, 1, bias=True)
    
    def get_go_frame(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Get the initial decoder input frame.
        
        Args:
            memory: Encoder memory
            
        Returns:
            torch.Tensor: Initial frame
        """
        batch_size = memory.size(0)
        go_frame = torch.zeros(
            batch_size, self.n_mel_channels, device=memory.device
        )
        return go_frame
    
    def initialize_decoder_states(self, memory: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Initialize decoder states.
        
        Args:
            memory: Encoder memory
            mask: Attention mask
            
        Returns:
            Dict containing decoder states
        """
        batch_size = memory.size(0)
        MAX_TIME = memory.size(1)
        
        attention_hidden = torch.zeros(
            batch_size, self.attention_rnn_dim, device=memory.device
        )
        attention_cell = torch.zeros(
            batch_size, self.attention_rnn_dim, device=memory.device
        )
        
        decoder_hidden = torch.zeros(
            batch_size, self.decoder_dim, device=memory.device
        )
        decoder_cell = torch.zeros(
            batch_size, self.decoder_dim, device=memory.device
        )
        
        attention_weights = torch.zeros(
            batch_size, MAX_TIME, device=memory.device
        )
        attention_weights_cat = torch.zeros(
            batch_size, 2, MAX_TIME, device=memory.device
        )
        
        processed_memory = self.attention_layer.memory_layer(memory)
        
        return {
            'attention_hidden': attention_hidden,
            'attention_cell': attention_cell,
            'decoder_hidden': decoder_hidden,
            'decoder_cell': decoder_cell,
            'attention_weights': attention_weights,
            'attention_weights_cat': attention_weights_cat,
            'processed_memory': processed_memory
        }
    
    def parse_decoder_outputs(self, mel_outputs: torch.Tensor, gate_outputs: torch.Tensor,
                            alignments: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse decoder outputs.
        
        Args:
            mel_outputs: Mel outputs
            gate_outputs: Gate outputs
            alignments: Attention alignments
            
        Returns:
            Tuple of parsed outputs
        """
        # Reshape mel outputs
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels
        )
        
        # Transpose mel outputs
        mel_outputs = mel_outputs.transpose(1, 2)
        
        # Gate outputs
        gate_outputs = gate_outputs.transpose(1, 2)
        gate_outputs = gate_outputs.squeeze(1)
        
        return mel_outputs, gate_outputs, alignments
    
    def decode(self, decoder_input: torch.Tensor, states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single decoder step.
        
        Args:
            decoder_input: Decoder input frame
            states: Decoder states
            
        Returns:
            Tuple of (mel output, gate output, attention weights, updated states)
        """
        # Prenet
        prenet_output = self.prenet(decoder_input)
        
        # Attention RNN
        attention_hidden, attention_cell = self.attention_rnn(
            torch.cat((prenet_output, states['attention_context']), -1),
            (states['attention_hidden'], states['attention_cell'])
        )
        
        # Attention mechanism
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, states['memory'], states['processed_memory'],
            states['attention_weights_cat'], states['mask']
        )
        
        # Decoder RNN
        decoder_input_rnn = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input_rnn,
            (states['decoder_hidden'], states['decoder_cell'])
        )
        
        # Linear projections
        decoder_output = torch.cat((decoder_hidden, attention_context), -1)
        mel_output = self.linear_projection(decoder_output)
        gate_output = self.gate_layer(decoder_output)
        
        # Update states
        states['attention_hidden'] = attention_hidden
        states['attention_cell'] = attention_cell
        states['decoder_hidden'] = decoder_hidden
        states['decoder_cell'] = decoder_cell
        states['attention_context'] = attention_context
        states['attention_weights'] = attention_weights
        
        # Update attention weights for location-sensitive attention
        attention_weights_cat = torch.cat([
            states['attention_weights'].unsqueeze(1),
            attention_weights.unsqueeze(1)
        ], dim=1)
        states['attention_weights_cat'] = attention_weights_cat
        
        return mel_output, gate_output, attention_weights, states
    
    def forward(self, memory: torch.Tensor, decoder_inputs: torch.Tensor,
                memory_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            memory: Encoder memory
            decoder_inputs: Decoder input frames
            memory_lengths: Memory lengths
            
        Returns:
            Tuple of (mel outputs, gate outputs, alignments)
        """
        # Create attention mask
        mask = memory_lengths.new_zeros(memory.size(0), memory.size(1), dtype=torch.bool)
        for i, length in enumerate(memory_lengths):
            mask[i, length:] = True
        
        # Initialize decoder states
        states = self.initialize_decoder_states(memory, mask)
        states['memory'] = memory
        states['mask'] = mask
        
        # Get initial frame
        decoder_input = self.get_go_frame(memory)
        
        # Decode
        mel_outputs, gate_outputs, alignments = [], [], []
        
        for i in range(decoder_inputs.size(1)):
            decoder_input = decoder_inputs[:, i, :]
            mel_output, gate_output, attention_weights, states = self.decode(
                decoder_input, states
            )
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        return mel_outputs, gate_outputs, alignments


class Postnet(nn.Module):
    """
    Postnet for mel-spectrogram refinement.
    """
    
    def __init__(self, n_mel_channels: int = 80, postnet_embed_dim: int = 512,
                 postnet_kernel_size: int = 5, postnet_n_convs: int = 5):
        """
        Initialize postnet.
        
        Args:
            n_mel_channels: Number of mel channels
            postnet_embed_dim: Postnet embedding dimension
            postnet_kernel_size: Postnet kernel size
            postnet_n_convs: Number of postnet convolutions
        """
        super().__init__()
        
        conv_layers = []
        
        # First convolution
        conv_layers.append(
            nn.Conv1d(n_mel_channels, postnet_embed_dim,
                     kernel_size=postnet_kernel_size, padding=2)
        )
        conv_layers.append(nn.BatchNorm1d(postnet_embed_dim))
        conv_layers.append(nn.Tanh())
        conv_layers.append(nn.Dropout(0.5))
        
        # Middle convolutions
        for i in range(postnet_n_convs - 2):
            conv_layers.append(
                nn.Conv1d(postnet_embed_dim, postnet_embed_dim,
                         kernel_size=postnet_kernel_size, padding=2)
            )
            conv_layers.append(nn.BatchNorm1d(postnet_embed_dim))
            conv_layers.append(nn.Tanh())
            conv_layers.append(nn.Dropout(0.5))
        
        # Last convolution
        conv_layers.append(
            nn.Conv1d(postnet_embed_dim, n_mel_channels,
                     kernel_size=postnet_kernel_size, padding=2)
        )
        conv_layers.append(nn.BatchNorm1d(n_mel_channels))
        conv_layers.append(nn.Dropout(0.5))
        
        self.convolutions = nn.Sequential(*conv_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input mel-spectrogram [batch, n_mels, time]
            
        Returns:
            torch.Tensor: Refined mel-spectrogram
        """
        return self.convolutions(x)


class Tacotron2(nn.Module):
    """
    Tacotron2 model for singing voice synthesis.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Tacotron2 model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        acoustic_config = config.acoustic_model
        
        # Components
        self.encoder = Encoder(
            vocab_size=config.text_processing.phoneme_vocab_size,
            encoder_dim=acoustic_config.encoder_dim,
            encoder_n_convs=acoustic_config.encoder_n_convs,
            encoder_conv_dim=acoustic_config.encoder_conv_dim
        )
        
        self.decoder = Decoder(
            n_mel_channels=config.audio.n_mel_channels,
            encoder_dim=acoustic_config.encoder_dim,
            decoder_dim=acoustic_config.decoder_rnn_dim,
            prenet_dim=acoustic_config.prenet_dim,
            attention_rnn_dim=acoustic_config.attention_rnn_dim,
            attention_dim=acoustic_config.attention_dim,
            attention_location_n_filters=acoustic_config.attention_location_n_filters,
            attention_location_kernel_size=acoustic_config.attention_location_kernel_size,
            max_decoder_steps=acoustic_config.max_decoder_steps,
            gate_threshold=acoustic_config.gate_threshold
        )
        
        self.postnet = Postnet(
            n_mel_channels=config.audio.n_mel_channels,
            postnet_embed_dim=acoustic_config.postnet_conv_dim,
            postnet_kernel_size=acoustic_config.postnet_kernel_size,
            postnet_n_convs=acoustic_config.postnet_n_convs
        )
    
    def parse_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse batch data.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Tuple of parsed tensors
        """
        text_padded = batch['text']
        input_lengths = batch['text_lengths']
        mel_padded = batch['mel']
        gate_padded = batch['gate']
        
        return text_padded, input_lengths, mel_padded, gate_padded
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Dict containing model outputs
        """
        text_padded, input_lengths, mel_padded, gate_padded = self.parse_batch(batch)
        
        # Encoder
        encoder_outputs, encoder_lengths = self.encoder(text_padded, input_lengths)
        
        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_padded, encoder_lengths
        )
        
        # Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'gate_outputs': gate_outputs,
            'alignments': alignments
        }
    
    def inference(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference mode.
        
        Args:
            text: Input text tensor
            
        Returns:
            Dict containing generated outputs
        """
        # Encoder
        encoder_outputs, encoder_lengths = self.encoder(text, torch.tensor([text.size(1)]))
        
        # Decoder (inference mode)
        mel_outputs, gate_outputs, alignments = self._inference_decode(encoder_outputs)
        
        # Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'gate_outputs': gate_outputs,
            'alignments': alignments
        }
    
    def _inference_decode(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference decoding.
        
        Args:
            encoder_outputs: Encoder outputs
            
        Returns:
            Tuple of generated outputs
        """
        # Initialize decoder states
        states = self.decoder.initialize_decoder_states(encoder_outputs)
        states['memory'] = encoder_outputs
        states['mask'] = None
        
        # Get initial frame
        decoder_input = self.decoder.get_go_frame(encoder_outputs)
        
        mel_outputs, gate_outputs, alignments = [], [], []
        
        for i in range(self.decoder.max_decoder_steps):
            mel_output, gate_output, attention_weights, states = self.decoder.decode(
                decoder_input, states
            )
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
            
            # Check for stopping condition
            if torch.sigmoid(gate_output) > self.decoder.gate_threshold:
                break
            
            # Use generated mel as next input
            decoder_input = mel_output
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        return mel_outputs, gate_outputs, alignments
