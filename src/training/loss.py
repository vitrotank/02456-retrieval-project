import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self, temperature = 0.05):
        super().__init__()

        self.temperature = temperature
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 1d torch tensor 
        """

        #  After normalization, every score is a cosine between −1 and 1. With a batch of 32, each question competes against 31 negatives,
        #  if we dont include TaU, it cannot give a rigth answer  a good probability, using a TaU below 1 helps to stretches the range of the values
        #  so softmasx becomes more confidetn 
        
        scores = torch.matmul(queries, documents.T) / self.temperature
        positive_scores = torch.diagonal(scores)
        denominator = torch.logsumexp(scores,1)
        return denominator - positive_scores