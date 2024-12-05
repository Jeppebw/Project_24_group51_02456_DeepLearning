"""Mamba model."""

from typing import Any, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.functional import gelu
from transformers import MambaConfig, MambaForCausalLM
from transformers.activations import ACT2FN
from models.embeddings import MambaEmbeddingsForCEHR

class MambaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        hidden_size: int = 16,  # Size of the hidden states from the base model - was 32
        classifier_dropout: float = 0.1,  # Dropout probability
        num_labels: int = 2,  # Number of classes for classification
        hidden_act: str = "gelu",  # Activation function
    ):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.act_fn = nn.ReLU()

        # Choose activation function
        self.hidden_act = ACT2FN.get(hidden_act, nn.ReLU())  # Default to ReLU if not in ACT2FN

    def forward(self, features, **kwargs):
        """Forward pass."""
        x = features  # Pooling is done by the forward pass
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits

############ Mamba model 1 #############

class EHRmamba(nn.Module):
    """Mamba model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 25, #Was 50, 32 for less param
        time_embeddings_size: int = 32,
        static_features_size: int = 8,
        num_measurements: int = 37,
        max_timesteps: int = 215,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 2, #Was 10, 8 for less param
        num_hidden_layers: int = 1, # was 2
        expand: int = 2,
        conv_kernel: int = 3, # was 4
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        use_mambapy: bool = False,
        num_labels: int = 2,  # Number of labels for classification
        classifier_dropout: float = 0.2,  # Dropout for the classification head - was 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.static_features_size = static_features_size
        self.num_measurements = num_measurements
        self.max_timesteps = max_timesteps
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.use_mambapy = use_mambapy

        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
            use_mambapy=self.use_mambapy,
        )
        
        # Embedding layer
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            num_measurements=self.num_measurements,
            max_timesteps=self.max_timesteps,
            static_features_size=self.static_features_size,
            time_embeddings_size=self.time_embeddings_size,
            hidden_dropout_prob=self.dropout_prob,
        )

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config)
        
        # Classification head
        self.classification_head = MambaClassificationHead(
            hidden_size=self.embedding_size,
            classifier_dropout=classifier_dropout,
            num_labels=num_labels,
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        time_series_data,
        static_data,
        time_array,
        #sensor_mask,
        #inputs: Tuple[
        #    torch.Tensor,  # Time series data
        #    torch.Tensor,  # Static data
        #    torch.Tensor,  # Time array
        #    Optional[torch.Tensor],  # Sensor mask
        #],
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ): #-> Union[Tuple[torch.Tensor, ...]]:
        """Forward pass for the model."""
       # time_series_data, static_data, time_array, sensor_mask = inputs
        
        #print("Shape of time_series_data:", time_series_data.shape)
        #print("Shape of Static:", static_data.shape)
        #print("Shape of time_array:", time_array.shape)
        #print("Shape of sensor_mask:", sensor_mask.shape)
        
        # Step 1: Embed the inputs
        inputs_embeds = self.embeddings(
            time_series_data=time_series_data,
            static_data=static_data,
            time_array=time_array,
            #sensor_mask=sensor_mask,
        )
        # In main model forward pass
        #print("Shape of inputs_embeds:", inputs_embeds.shape)

        # Step 2: Process through Mamba model
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        
        # Step 3: Apply classification head
        # Assuming the last hidden state from Mamba model is the pooled representation
        pooled_output = torch.mean(outputs.hidden_states[-1], dim=1)  # Global average pooling
        #print(f"Shape of pooled_output: {pooled_output.shape}")  # Ensure this is [batch_size, hidden_size]

        
        logits = self.classification_head(pooled_output)
        #print("Shape of logits:", logits.shape)
        
        # Calculate loss if labels are provided
        if labels is not None:
            print(f"Shape of labels: {labels.shape}")
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}

        return logits

    #def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
    #    """Train model on training dataset."""
    #    inputs = (
    #        batch["concept_ids"],
    #        batch["type_ids"],
    #        batch["time_stamps"],
    #        batch["ages"],
    #        batch["visit_orders"],
    #        batch["visit_segments"],
    #    )
    #    labels = batch["labels"]
    #
    #    # Ensure use of mixed precision
    #    with autocast():
    #        loss = self(
    #            inputs,
    #            labels=labels,
    #            return_dict=True,
    #        ).loss
    #
    #    (current_lr,) = self.lr_schedulers().get_last_lr()
    #    self.log_dict(
    #        dictionary={"train_loss": loss, "lr": current_lr},
    #        on_step=True,
    #        prog_bar=True,
    #        sync_dist=True,
    #    )
    #    return loss

    #def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
    #    """Evaluate model on validation dataset."""
    #    inputs = (
    #        batch["concept_ids"],
    #        batch["type_ids"],
    #        batch["time_stamps"],
    #        batch["ages"],
    #        batch["visit_orders"],
    #        batch["visit_segments"],
    #    )
    #    labels = batch["labels"]
    #
    #    # Ensure use of mixed precision
    #    with autocast():
    #        loss = self(
    #            inputs,
    #            labels=labels,
    #            return_dict=True,
    #        ).loss
    #
    #    (current_lr,) = self.lr_schedulers().get_last_lr()
        #self.log_dict(
        #    dictionary={"val_loss": loss, "lr": current_lr},
        #    on_step=True,
        #    prog_bar=True,
        #    sync_dist=True,
        #)
    #    return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
