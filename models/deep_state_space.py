import torch
import torch.nn as nn
import torch.nn.functional as F

class StateSpaceModelForClassification(nn.Module):
    def __init__(self, state_dim, num_classes, control_dim=None, hidden_dim=64, num_heads=8, noise_std=0.1):
        """
        Extended state-space model for classification using Transformer-based state transition.

        Args:
            state_dim (int): Dimension of the latent state.
            num_classes (int): Number of output classes.
            control_dim (int): Dimension of control input (optional).
            hidden_dim (int): Dimension of hidden layers in the observation model.
            num_heads (int): Number of attention heads for the Transformer.
            noise_std (float): Standard deviation of the Gaussian noise in the state transition.
        """
        super().__init__()
        
        # State transition: Transformer model for sequence-to-sequence mapping
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=4, 
            num_decoder_layers=4, 
            dim_feedforward=256,
            dropout=0.1
        )
        
        # Control model: Optional, integrates control input into state
        if control_dim is not None:
            self.control_model = nn.Linear(control_dim, hidden_dim)
        else:
            self.control_model = None

        # Observation model: Output logits for classification
        self.observation_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # Output number of classes, no softmax here
        )
        
        # Stochasticity in state transition
        self.noise_std = noise_std

    def forward(self, state, control_input=None):
        """
        Forward pass for a single time step with Transformer-based state transition.
        
        Args:
            state (torch.Tensor): Latent state tensor of shape (batch_size, state_dim).
            control_input (torch.Tensor, optional): Control input tensor.

        Returns:
            next_state (torch.Tensor): Next latent state tensor.
            logits (torch.Tensor): Output logits for classification.
        """
        # Add noise to the state
        noisy_state = state + torch.randn_like(state) * self.noise_std
        
        # Embed the state for transformer input
        state_emb = self.state_embedding(noisy_state)

        # If control is provided, integrate it into the transformer input
        if self.control_model is not None and control_input is not None:
            control_emb = self.control_model(control_input)
            state_emb += control_emb
        
        # Reshape to fit Transformer input shape: (batch_size, seq_len, feature_dim)
        state_emb = state_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Transformer expects a sequence, so we can pass the state as both the source and target
        transformer_output = self.transformer(state_emb, state_emb)
        
        # Extract the transformed state
        next_state = transformer_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Observation model (logits)
        logits = self.observation_nn(next_state)
        
        return next_state, logits

