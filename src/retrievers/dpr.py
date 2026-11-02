from src.retrievers.models.encoder import TransformerEncoder
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


