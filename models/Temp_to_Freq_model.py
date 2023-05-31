"""
Temporal to Frequency label model
Based on the Transforemr Encoder-Decoder model with the following modifications:
1. The encoder is formed by a series of 1D conv-Norm-Transformer blocks
2. The decoder is formed by a series of Tramsformer decoder blocks with num of query = time steps
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath
from .positional_embedding import build_position_encoding


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "leaky_relu":
        return F.leaky_relu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransEncoder_Conv1d_Act_block(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=64):
        super(TransEncoder_Conv1d_Act_block, self).__init__()

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = Transformer_Encoder(num_layers=num_layers, norm=encoder_norm, d_model=d_model,
                                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           drop_path=drop_path, activation=activation, 
                                           normalize_before=normalize_before)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length, 
                                kernel_size=kernel, stride=1, padding=kernel // 2 - 1)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        # src: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.encoder(src, src_key_padding_mask, pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self, num_blocks=3, num_layers=4, d_model=512, nhead=8, 
                 dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=32) -> None:
        super(Encoder, self).__init__()

        block_params = {
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "drop_path": drop_path,
            "activation": activation,
            "normalize_before": normalize_before,
            "kernel": kernel,
            "sequence_length": sequence_length
        }

        block_list = []
        for _ in range(num_blocks):
            block_list.append(TransEncoder_Conv1d_Act_block(**block_params))
            block_params["d_model"] /= 2
        
        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]

        self.norm = _get_activation_fn("tanh")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: [B, L, Embedding] 
            L is the sequence length; Embedding is the embedding dimension; B is the batch size
        """
        x = src

        for block in self.blocks:
            x = block(x, src_key_padding_mask, pos_embed)
        return self.norm(x)


class TransDecoder_Conv1d_Act_block(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=64) -> None:
        super(TransDecoder_Conv1d_Act_block, self).__init__()

        deoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = Transformer_Decoder(num_layers=num_layers, norm=deoder_norm, d_model=d_model,
                                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           drop_path=drop_path, activation=activation,
                                           normalize_before=normalize_before)
        self.conv1d = nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length,
                                kernel_size=kernel, stride=1, padding=kernel // 2 - 1)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) -> Tensor:
        # tgt: [B, L, Embedding] 
        # L is the sequence length; Embedding is the embedding dimension; B is the batch size

        x = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                         pos_embed, query_pos_embed)
        x = self.conv1d(x)
        return self.activation(x)


class Decoder(nn.Module):
    def __init__(self, num_blocks=3, num_layers=4, d_model=512, nhead=8, 
                 dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", normalize_before=True,
                 kernel=7, sequence_length=32) -> None:
        super(Decoder, self).__init__()

        block_params = {
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "drop_path": drop_path,
            "activation": activation,
            "normalize_before": normalize_before,
            "kernel": kernel,
            "sequence_length": sequence_length
        }

        block_list = [TransDecoder_Conv1d_Act_block(**block_params) 
                      for _ in range(num_blocks)]

        self.blocks = nn.ModuleList(block_list)
        self.embed_dim = block_params["d_model"]

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            tgt: [B, L, Embedding] 
            L is the sequence length; Embedding is the embedding dimension; B is the batch size
        """
        x = tgt

        for block in self.blocks:
            x = block(x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                      pos_embed, query_pos_embed)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: dict = None, pos_type: str = "sine") -> None:
        super(Transformer, self).__init__()
        assert cfg is not None, "cfg is None"
        self.encoder_cfg = {
            "num_blocks": cfg["num_encoder_blocks"],
            "num_layers": cfg["num_encoder_layers"],
            "d_model": cfg["encoder_embed_model"],
            "nhead": cfg["encoder_nhead"],
            "dropout": cfg["dropout"],
            "drop_path": cfg["drop_path"],
            "sequence_length": cfg["encoder_sequence_length"],
        }

        self.decoder_cfg = {
            "num_blocks": cfg["num_decoder_blocks"],
            "num_layers": cfg["num_decoder_layers"],
            "d_model": cfg["decoder_embed_model"],
            "nhead": cfg["decoder_nhead"],
            "dropout": cfg["dropout"],
            "drop_path": cfg["drop_path"],
            "sequence_length": cfg["num_queries"],
        }

        self.encoder = Encoder(**self.encoder_cfg)
        self.decoder = Decoder(**self.decoder_cfg)

        self.query_embed = nn.Embedding(cfg["num_queries"], self.decoder_cfg["d_model"])
        
        self.encoder_pos = build_position_encoding(type=pos_type, 
                                                   d_model=self.encoder_cfg["d_model"])
        self.decoder_pos = build_position_encoding(type=pos_type,
                                                   d_model=self.encoder_cfg["d_model"])
        self.query_pos = build_position_encoding(type=pos_type,
                                                 d_model=self.decoder_cfg["d_model"])
        
        self.classify_head = nn.Parameter(torch.randn(cfg["num_queries"], 
                                                      cfg["decoder_embed_model"], 
                                                      cfg["num_classes"]))
        
    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None) -> Tensor:
        src_pos_embed = self.encoder_pos(src)
        memory = self.encoder(src, src_mask, src_pos_embed)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query_pos_embed = self.query_pos(query_embed)

        tgt = torch.zeros_like(query_embed)
        decoder_pos_embed = self.decoder_pos(memory)
        # hs: [B, num_queries, decoder_embed_model]
        hs = self.decoder(tgt, memory, pos_embed=decoder_pos_embed, query_pos_embed=query_pos_embed)

        pred_cls = torch.einsum("bnd,ndc->bnc", hs, self.classify_head)
        return pred_cls
        