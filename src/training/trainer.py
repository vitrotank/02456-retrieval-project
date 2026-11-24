import torch
import torch.nn as nn
from src.training.loss import DPRLoss

class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(self, 
              query_encoder: nn.Module, 
              document_encoder: nn.Module, 
              train_loader, 
              tokenizer, 
              epochs=3, 
              lr=1e-4):
        device = next(query_encoder.parameters()).device  # Get the device of the model parameters

        parameters = (list(query_encoder.parameters()) 
                      + list(document_encoder.parameters())
                      )
        optimizer = torch.optim.AdamW(parameters, lr=lr)
        self.criterion.to(device)

        positive_passges = {}
        dataset = train_loader.dataset
        for question, texts in zip(dataset.questions, dataset.texts):
            positive_passges.setdefault(question, set()).update(texts)

        for epoch in range(epochs):
            query_encoder.train()
            document_encoder.train()

            total_loss = 0.0
            total_pairs = 0

            for questions, passages in train_loader:
                batch_size = len(questions)
                if batch_size < 2:
                    continue  # Skip batches that are too small to form pairs

                if any(passages[j] in positive_passges[questions[i]] 
                       for i in range(batch_size) 
                       for j in range(batch_size) 
                       if i != j):
                    continue  # Skip batches with positive pairs

                query_tokens = tokenizer(
                    list(questions),
                    padding=True,
                    truncation=True,
                    max_length=min(64, query_encoder.max_len),
                    return_tensors="pt",
                ).to(device)

                passage_tokens = tokenizer(
                    list(passages),
                    padding=True,
                    truncation=True,
                    max_length=min(256, document_encoder.max_len),
                    return_tensors="pt",
                ).to(device)

                optimizer.zero_grad()

                query_embeddings=query_encoder(
                    query_tokens['input_ids'],
                    query_tokens['attention_mask'],
                )

                passage_embeddings=document_encoder(
                    passage_tokens['input_ids'],
                    passage_tokens['attention_mask'],
                ) 

                # Compute loss
                loss = self.criterion(query_embeddings, passage_embeddings).mean()  # Mean over the batch
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*batch_size
                total_pairs += batch_size

            if total_pairs == 0:
                print(f"Epoch {epoch + 1}: no valid batches (all skipped)")
                continue  # Avoid division by zero

            print(f"Epoch {epoch + 1}: loss={total_loss / total_pairs:.4f}")