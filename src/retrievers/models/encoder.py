import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, vocab_size=30522, max_len=512, pooling_type='cls', dropout=0.1, pad_token_id=0):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.setup_model()
        

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim, padding_idx=self.pad_token_id)
        self.position_embedding = nn.Embedding(num_embeddings=self.max_len, embedding_dim=self.emb_dim)

        self.model = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feed_forward_dim,
                dropout=self.dropout,
                batch_first=True,
                activation='relu'

            ),
            num_layers=self.n_layers
        )
        return self.model

    def pool(self, x, attention_mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.pooling_type == 'cls':
            return x[:, 0, :]  # Return the first token (CLS token) for each sequence
        
        elif self.pooling_type == 'max':
            max_masked = x.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
            return torch.max(max_masked, dim=1).values  # Return the maximum value along the sequence_length dimension for each sequence
        
        elif self.pooling_type == 'mean':
            masked_mean = torch.sum(x * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            return masked_mean  # Return the mean value along the sequence_length dimension for each sequence

        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}. Choose from 'cls', 'max', or 'mean'.")
        
    def forward(self, input_ids, attention_mask):
        """
        Run a forward pass on the model
        """
        x = self.token_embedding(input_ids) + self.position_embedding(torch.arange(input_ids.size(1), device=input_ids.device))
        x = self.model(x, src_key_padding_mask=(attention_mask == 0) )
        x = self.pool(x, attention_mask)  # Pool the output embeddings
        return x
