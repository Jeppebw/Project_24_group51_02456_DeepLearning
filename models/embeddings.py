"""Embedding layers for the models."""

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import MambaConfig


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> Any:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(
        self,
        visit_order_size: int,
        embedding_size: int,
    ):
        super().__init__()
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> Any:
        """Apply visit embedding to the input visit segments."""
        return self.embedding(visit_segments)


class ConceptEmbedding(nn.Module):
    """Embedding layer for event concepts."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        padding_idx: Optional[int] = None,
    ):
        super(ConceptEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
            padding_idx=padding_idx,
        )

    def forward(self, inputs: torch.Tensor) -> Any:
        """Apply concept embedding to the input concepts."""
        return self.embedding(inputs)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""

    def __init__(self, embedding_size: int, max_len: int):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, visit_orders: torch.Tensor) -> Any:
        """Apply positional embedding to the input visit orders."""
        first_visit_concept_orders = visit_orders[:, 0:1]
        normalized_visit_orders = torch.clamp(
            visit_orders - first_visit_concept_orders,
            0,
            self.pe.size(0) - 1,
        )
        return self.pe[normalized_visit_orders]

class MambaEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from concept, token_type, etc., embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: MambaConfig,
        num_measurements: int = 37,
        max_timesteps: int = 215,
        static_features_size: int = 8,
        time_embeddings_size: int = 32,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initiate wrapper class for embeddings used in Mamba CEHR classes."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_measurements = num_measurements

        # Embeddings for the time series measurements
        self.measurement_embeddings = nn.Linear(
            num_measurements, config.hidden_size
        )  # Project measurements to hidden size

        # Static data embeddings
        self.static_embeddings = nn.Linear(
            static_features_size, config.hidden_size
        )  # Project static features to hidden size

        # Positional and temporal embeddings
        self.positional_embeddings = nn.Embedding(max_timesteps, config.hidden_size)
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=False,  # Change as per your time array use case
        )
        
        # Combine embeddings and scale back
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size * 2 + time_embeddings_size, config.hidden_size
        )  # Combine measurement, positional, and static embeddings        
        
        # Part of the old EHRMamba code
        
        #self.age_embeddings = TimeEmbeddingLayer(
        #    embedding_size=time_embeddings_size,
        #)
        #
        #self.scale_back_concat_layer = nn.Linear(
        #    config.hidden_size + 2 * time_embeddings_size,
        #    config.hidden_size,
        #)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # End copy

    def forward(
        self,
        time_series_data: torch.Tensor,
        static_data: torch.Tensor,
        time_array: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Return the final embeddings of concept ids.

        Parameters
        ----------
        time_series_data : torch.Tensor
            Time series input data of shape (batch_size, num_measurements, timesteps).
        static_data : torch.Tensor
            Static input data of shape (batch_size, static_features_size).
        time_array : torch.Tensor
            Time array of shape (batch_size, timesteps).
        sensor_mask : torch.Tensor, optional
            Sensor mask for valid/invalid data of shape (batch_size, num_measurements, timesteps).
        """
         # (1) Measurement embeddings (time series data)
        # Input: (batch_size, num_features, timesteps)
        # Linear operates on the num_features axis
        batch_size, num_features, timesteps = time_series_data.shape
        ts_embeds = self.measurement_embeddings(time_series_data.permute(0, 2, 1))  
        # Output shape: (batch_size, timesteps, hidden_size)

        #print(f"Shape of time-series embedded: {ts_embeds.shape}")
        # Apply sensor mask if provided
        if sensor_mask is not None:
            # Ensure sensor_mask is broadcastable with ts_embeds
            sensor_mask = sensor_mask.unsqueeze(-1)  # Shape: (batch_size, num_features, timesteps, 1)
            ts_embeds = ts_embeds * sensor_mask.permute(0, 2, 1, 3).squeeze(-1)  # Shape: (batch_size, timesteps, hidden_size)

        # (2) Static data embeddings
        # Input: (batch_size, static_features_size)
        static_embeds = self.static_embeddings(static_data)
        static_embeds = static_embeds.unsqueeze(1).expand(-1, timesteps, -1)  
        # Broadcast over timesteps: (batch_size, timesteps, hidden_size)

        # (3) Positional embeddings for timesteps
        position_ids = torch.arange(timesteps, dtype=torch.long, device=ts_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.positional_embeddings(position_ids)  
        # Output shape: (batch_size, timesteps, hidden_size)

        # (4) Time embeddings
        time_embeds = self.time_embeddings(time_array)  
        # Output shape: (batch_size, timesteps, time_embeddings_size)

        # (5) Combine embeddings
        # Concatenate measurement, static, and positional embeddings
        combined_embeds = torch.cat((ts_embeds, static_embeds, time_embeds), dim=-1)
        # Shape after concatenation: (batch_size, timesteps, hidden_size * 3)

        #print(f"Shape of combined embeddings: {combined_embeds.shape}")

        # Scale back to hidden size
        combined_embeds = self.tanh(self.scale_back_concat_layer(combined_embeds))

        # (6) Apply dropout and layer normalization
        combined_embeds = self.dropout(combined_embeds)
        combined_embeds = self.LayerNorm(combined_embeds)

        return combined_embeds