import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import SetEncoder
from models.modules import GELU, LayerNorm


class TagPredictor(nn.Module):
    def __init__(self, args, question_info_dict, answerer_info_dict, tag_to_idx, device):
        super(TagPredictor, self).__init__()
        self.args = args
        self.device = device
        self.num_tag = len(tag_to_idx)
        self.tPAD = self.num_tag

        self.embedding_dim = args.embedding_dim
        # last_idx points the padding in the tag_embedding
        self.tag_embedding = nn.Embedding(self.num_tag+1, self.embedding_dim)
        self.dropout = nn.Dropout(self.args.dropout)

        self.layer_type = args.layer_type
        # question -> answerer
        question_encoder = []
        for _ in range(self.args.num_hidden_layers):
            question_encoder.append(SetEncoder(self.layer_type,
                                               self.embedding_dim,
                                               self.args.num_heads,
                                               self.args.dropout))
        # answerer -> question
        answerer_encoder = []
        for _ in range(self.args.num_hidden_layers):
            answerer_encoder.append(SetEncoder(self.layer_type,
                                               self.embedding_dim,
                                               self.args.num_heads,
                                               self.args.dropout))
        self.question_encoder = nn.ModuleList(question_encoder)
        self.answerer_encoder = nn.ModuleList(answerer_encoder)
        self.norm = LayerNorm(self.embedding_dim, eps=1e-6)
        self.output_layer = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        """
        Inputs
            question_by_tag_ids: (N (batch_size), A (max_answerers), Q (max_questions), T (max_tags))
            tag_ids: (N, C)
        Outputs
            logits: torch.tensor: (N, C)
        """
        question_by_tag_ids, tag_ids = x
        N, A, Q, T = question_by_tag_ids.shape
        E = self.embedding_dim

        # mask: 1 represents the padded instance
        tag_mask = (question_by_tag_ids == self.tPAD).float()
        question_mask = (torch.sum(tag_mask, -1) == T).float()
        answerer_mask = (torch.sum(question_mask, -1) == Q).float()

        # tags -> question
        x = self.tag_embedding(question_by_tag_ids.reshape(-1)).reshape(N, A, Q, T, E)
        # average over the valid tags
        x = (1 - tag_mask).unsqueeze(-1) * x
        x = torch.mean(x, -2)

        # questions -> answerer
        question_attn_mask = (1 - question_mask.reshape(-1, Q))
        question_attn_mask = question_attn_mask.unsqueeze(-1).matmul(question_attn_mask.unsqueeze(1))
        question_attn_mask = (1 - question_attn_mask)
        x = x.reshape(-1, Q, E)
        for layer in self.question_encoder:
            x = layer(x, question_attn_mask)
        x = x.reshape(N, A, Q, E)
        # average over the valid questions
        x = (1 - question_mask).unsqueeze(-1) * x
        x = torch.mean(x, -2)

        # answerers -> question
        answerer_attn_mask = (1 - answerer_mask).unsqueeze(-1).matmul(1 - answerer_mask.unsqueeze(1))
        answerer_attn_mask = (1 - answerer_attn_mask)
        for layer in self.question_encoder:
            x = layer(x, answerer_attn_mask)
        x = (1 - answerer_mask).unsqueeze(-1) * x
        # average over the valid answerers
        x = torch.mean(x, -2)
        x = self.norm(x)

        # output-layer
        tag_embedding = self.tag_embedding(tag_ids)  # [N, E, 1]
        x = tag_embedding.matmul(x.unsqueeze(-1)).squeeze(-1)

        return x
