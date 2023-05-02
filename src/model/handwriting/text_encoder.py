import torch
import torch.nn as nn


class TextEncoder(nn.Module):

    def __init__(self, char_embedding, pos_encoding, config):
        super(TextEncoder, self).__init__()

        self.char_embedding = char_embedding
        self.pos_encoding = pos_encoding
        self.dropout = nn.Dropout(p=config.transformer_encoder['dropout'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.char_embedding_dim,
                                                   nhead=config.transformer_encoder['num_heads'],
                                                   dropout=config.transformer_encoder['dropout'],
                                                   dim_feedforward=config.transformer_encoder['ffn'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=config.transformer_encoder['num_layers'])

        # self.linear = nn.Linear(2048, 512)

    def forward(self, src, src_key_padding_mask, reformer_model=None):
        """
        ref_src_key_padding_mask = (~src_key_padding_mask).to(torch.long)
        src = reformer_model(input_ids=src, attention_mask=ref_src_key_padding_mask)
        # src = src.last_hidden_state
        src = src[0]
        src = nn.functional.normalize(src, p=2, dim=-1)
        src = self.linear(src)
        return src
        """
        # print('#'*10)
        # print(src)
        # print(src.shape)
        src = self.dropout(self.char_embedding(src) + self.pos_encoding(src))
        src = src.permute(1, 0, 2)
        src = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        src = src.permute(1, 0, 2)
        # print(src.shape)
        return src

