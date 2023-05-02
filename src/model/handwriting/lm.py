import torch
import torch.nn as nn

from src.data.handwriting.handwriting_dataset import DatasetHelper


class ARLM(nn.Module):

    def __init__(self, char_embedding, pos_encoding, config, vocab_size, device):
        super(ARLM, self).__init__()

        self.char_embedding = char_embedding
        self.device = device
        self.pos_encoding = pos_encoding
        self.dropout = nn.Dropout(p=config.transformer_encoder['dropout'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.char_embedding_dim,
                                                   nhead=config.transformer_encoder['num_heads'],
                                                   dropout=config.transformer_encoder['dropout'],
                                                   dim_feedforward=config.transformer_encoder['ffn'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=config.transformer_encoder['num_layers'])

        self.linear = nn.Linear(config.char_embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, src_mask, src_key_padding_mask):
        
        src = self.dropout(self.char_embedding(src) + self.pos_encoding(src))
        src = src.permute(1, 0, 2)
        src_mask = src_mask[0]
        src = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        src = src.permute(1, 0, 2)
        src = self.linear(src)
        return src

    def generate(self, sos_idx, eos_idx, max_char_len):

        result = [sos_idx]

        for i in range(max_char_len):

            tgt = torch.LongTensor([result]).to(self.device)
            tgt_mask = DatasetHelper.get_mask(i+1).to(self.device)
            tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

            tgt = tgt.permute(1, 0, 2)

            output = self.transformer_encoder(
                src=tgt,
                mask=tgt_mask,  # to avoid looking at the future tokens (the ones on the right)
                src_key_padding_mask=None  # avoid looking on padding of the src
            )

            output = output.permute(1, 0, 2)
            output = self.linear(output)
            output = self.softmax(output)
            output = output[0][-1]  # the last timestep

            values, indices = output.max(dim=0)
            pred_token = indices.item()
            result.append(pred_token)

            if pred_token == eos_idx:
                break

        return result

    def step(self, tgt, i):
        bs, seq_len = tgt.shape

        tgt_mask = DatasetHelper.get_mask(i+1).to(self.device)
        tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

        tgt = tgt.permute(1, 0, 2)

        output = self.transformer_encoder(
            src=tgt,
            mask=tgt_mask,
            src_key_padding_mask=None,  # to avoid working on padding
        )

        output = output.permute(1, 0, 2)

        output = output[:,i,:]

        output = self.linear(output)
        return output
        # output = self.softmax(output)

        # values, indices = output.max(dim=-1)
        # indices = indices.unsqueeze(-1)
        # result = torch.cat([result, indices],dim=-1)

