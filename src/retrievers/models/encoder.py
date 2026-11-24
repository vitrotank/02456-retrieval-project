import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class EncoderLayer(nn.Module):
    def __init__(
            self, emb_dim: int, hidden_dim: int, n_heads: int, feed_forward_dim, dropout: float = 0.1
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # If we want to try other parameters in the testing is better to assure that 
        # the dimenssions of each head are from the division of the (vector dimension) / (number of heads)
        # So we it can be tested with different hyperparameters (n_head - emb_dim)
        # Now changing hidden_dim will not have any effect

        assert emb_dim % n_heads == 0 , f"emb_dim({emb_dim}) must be divisible by n_head({n_heads})"
        self.head_dim = emb_dim // n_heads
        self.feed_forward_dim = feed_forward_dim    
        self.dropout = dropout

        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        self.output = nn.Linear(self.emb_dim, self.emb_dim)

        self.linear1 = nn.Linear(self.emb_dim, self.feed_forward_dim)
        self.linear2 = nn.Linear(self.feed_forward_dim, self.emb_dim)

        self.layer_norm1 = nn.LayerNorm(self.emb_dim)
        self.layer_norm2 = nn.LayerNorm(self.emb_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, attention_mask):
        batch_size, seq_length, embedding_size = x.size()

        q = self.q(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))  # Mask padding tokens
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attended_values = self.dropout_layer(attention_weights) @ v
        attended_values = attended_values.transpose(1, 2).reshape(batch_size, seq_length, embedding_size)

        x = self.layer_norm1(x + self.dropout_layer(self.output(attended_values)))
        
        ff = self.dropout_layer(F.relu(self.linear1(x)))
        ff = self.linear2(ff)
        x = self.layer_norm2(x + self.dropout_layer(ff))  

        return x

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

        self.model = nn.ModuleList([
            EncoderLayer(
                emb_dim=self.emb_dim,
                n_heads=self.n_heads,
                hidden_dim=self.hidden_dim,
                feed_forward_dim=self.feed_forward_dim,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
            ])
        
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
        positions = torch.arange(input_ids.size(1), device=input_ids.device)

        x = (self.token_embedding(input_ids) + 
             self.position_embedding(positions)
             )

        for layer in self.model:
            x = layer(x, attention_mask)

        #return self.pool(x, attention_mask)
        # Normalize the vector so the scale is 1 - this will help in softmax
        return F.normalize(self.pool(x, attention_mask), dim=-1)
