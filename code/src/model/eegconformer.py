import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        bsz, q_len, n_heads, d_k = queries.shape
        scale = self.scale or (1.0 / np.sqrt(d_k))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        bsz, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        h = self.n_heads

        q = self.query_projection(queries).view(bsz, q_len, h, -1)
        k = self.key_projection(keys).view(bsz, k_len, h, -1)
        v = self.value_projection(values).view(bsz, k_len, h, -1)

        out, attn = self.inner_attention(q, k, v, attn_mask, tau=tau, delta=delta)
        out = out.view(bsz, q_len, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class ShallowNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super().__init__()
        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 25), (1, 1)),
            nn.Conv2d(d_model, d_model, (c_in, 1), (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 4)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.shallow_net(x)
        x = self.projection(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return x


class EEGConformerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention

        self.enc_embedding = ShallowNetEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, configs.seq_len, configs.enc_in)
            feat = self.enc_embedding(dummy)
            feat, _ = self.encoder(feat, attn_mask=None)
            self.n_fc_in = feat.reshape(1, -1).shape[1]

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.n_fc_in, configs.num_class)

    def supervised(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        return self.supervised(x_enc, x_mark_enc)
