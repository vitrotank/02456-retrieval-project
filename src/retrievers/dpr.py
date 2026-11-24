from src.retrievers.models.encoder import TransformerEncoder
from src.data_utils.dataset import DocumentDataset
import torch
from transformers import AutoTokenizer


class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
        self.__load_models()

    @torch.no_grad()
    def encodetexts(self, texts, encoder, max_length, batch_size=64):
        encoder.eval()
        embeddings = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            tokens = self.tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            embeddings.append(encoder(input_ids, attention_mask))
        return torch.cat(embeddings, dim=0)

    def BuildIndex(self, documents, batch_size=64):
        # Call this after training
        self.documents = DocumentDataset(documents).GetDocuments()
        texts = [document.GetText() for document in self.documents]
        # One vector per document, same order as self.documents
        self.document_embeddings = self.encodetexts(texts, self.document_encoder, max_length=256, batch_size=batch_size)

    @torch.no_grad()
    def Lookup(self, queries, k, batch_size=256):
        k = min(k, len(self.documents))
        results = []
        # Score queries in chunks so the score matrix stays small
        for start in range(0, len(queries), batch_size):
            chunk = queries[start:start + batch_size]
            query_embeddings = self.encodetexts(chunk, self.query_encoder, max_length=64)
            scores = query_embeddings @ self.document_embeddings.T
            top_indices = torch.topk(scores, k, dim=1).indices.tolist()
            results.extend([[self.documents[i] for i in row] for row in top_indices])
        return results

    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        self.query_encoder = TransformerEncoder(
            n_layers=self.cfg.model.n_layers,
            n_heads=self.cfg.model.n_heads,
            emb_dim=self.cfg.model.emb_dim,
            hidden_dim=self.cfg.model.hidden_dim,
            feed_forward_dim=self.cfg.model.feed_forward_dim,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            pooling_type=self.cfg.model.pooling_type,
        ).to(self.device)

        self.document_encoder = TransformerEncoder(
            n_layers=self.cfg.model.n_layers,
            n_heads=self.cfg.model.n_heads,
            emb_dim=self.cfg.model.emb_dim,
            hidden_dim=self.cfg.model.hidden_dim,
            feed_forward_dim=self.cfg.model.feed_forward_dim,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            pooling_type=self.cfg.model.pooling_type,
        ).to(self.device)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(torch.load(query_weights_path, map_location=self.device))

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            self.document_encoder.load_state_dict(torch.load(document_weights_path, map_location=self.device))


