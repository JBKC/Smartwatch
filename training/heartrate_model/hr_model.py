'''
Model for estimating heart rate
Combines convolution of time window batches & attention across adjacent time windows
Called from hr_model_run.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import psutil


class ConvBlock(nn.Module):
    '''
    Single causal temporal convolution block
    Each block contains 3 convolutional layers
    '''

    def __init__(self, in_channels, n_filters, pool_size, kernel_size=5, dilation=2, dropout=0.5):

        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_filters = n_filters

        # conv block = [conv layer + RELU] * 3 + average pooling + dropout
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation
                ),
                nn.ReLU()
            ) for i in range(3)]
        )

        self.pool = nn.AvgPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        '''
        Pass data through the convolution block
        :param x: shape (batch_size, n_channels, sequence_length)
        :return x: shape (batch_size, 64, 16)
        '''

        for conv in self.conv_layers:
            # causal padding
            padding_size = self.dilation * (self.kernel_size - 1)

            # manually apply padding
            pad_tensor = torch.zeros(x.size(0), x.size(1), padding_size, device=x.device, dtype=x.dtype)
            x = torch.cat((pad_tensor, x), dim=2)

            # Apply convolution
            x = conv(x)

        # Apply pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)

        return x

class TemporalConvolution(nn.Module):
    '''
    Pass data through series of convolution blocks
    '''

    def __init__(self):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=1, n_filters=32, pool_size=4)
        self.conv_block2 = ConvBlock(in_channels=32, n_filters=48, pool_size=2)
        self.conv_block3 = ConvBlock(in_channels=48, n_filters=64, pool_size=2)

    def forward(self, x_cur, x_prev):
        '''
        Pass x_cur and x_prev in parallel through the same convolution blocks (weight sharing)
        :param x_cur, x_prev: shape (batch_size, n_channels, sequence_length)
        :return x_cur, x_prev: shape (batch_size, n_filters, embed_dim)
        '''

        x_cur = self.conv_block1(x_cur)

        x_prev = self.conv_block1(x_prev)

        x_cur = self.conv_block2(x_cur)
        x_prev = self.conv_block2(x_prev)

        x_cur = self.conv_block3(x_cur)
        x_prev = self.conv_block3(x_prev)

        return x_cur, x_prev



class AttentionModule(nn.Module):
    '''
    Cross-Attention module
    Input is the output of the convolutional blocks
    '''

    def __init__(self, n_embd=16, n_heads=4):
        super().__init__()

        # single attention module
        self.attention = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_heads, batch_first=True)

    def forward(self, query, key, value):
        '''
        :param query: x_prev, shape (batch_size, n_filters, embed_dim)
        :param key: x_cur, shape (batch_size, n_filters, embed_dim)
        :param value: x_cur
        :return: output of the attention formula
        '''

        out, _ = self.attention(query=query, key=key, value=value)

        return out


class TemporalAttentionModel(nn.Module):
    '''
    Attention architecture build
    '''

    def __init__(self, n_embd=16):
        super().__init__()

        self.convolution = TemporalConvolution()
        self.attention = AttentionModule()
        self.ln = nn.LayerNorm(normalized_shape=n_embd)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.125)
        self.fc2 = nn.Linear(in_features=256, out_features=2)

    def gaussian(self, x):
        '''
        :param x: estimated mu and standard deviation of HR for each window - shape (batch_size, 2)
        :return: Gaussian distribution of given inputs
        '''

        mu = x[:, 0]
        sigma = x[:, -1]
        sigma = 0.1 + F.softplus(sigma)

        return Normal(loc=mu, scale=sigma)


    def forward(self, x_cur, x_prev):
        '''
        :param x_cur: shape (batch_size, n_channels, sequence_length)
        :param x_prev: shape (batch_size, n_channels, sequence_length)
        :return out: probability distribution over output
        '''

        # temporal convolution
        x_cur, x_prev = self.convolution(x_cur, x_prev)

        # attention with residual connection: query = x_prev, key = value = x_cur
        x = x_cur + self.attention(x_prev, x_cur, x_cur)

        # layer normalisation over embedding dimension
        x = self.ln(x)

        # fully connected layers & dropout
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))

        # pass through probabilistic layer - this is the part not in SubModel
        out = self.gaussian(x)

        return out



class SubModel(TemporalAttentionModel):
    '''
    Inherit from TemporalAttentionModel but exclude final probability layer
    Used for error classifier
    '''

    def __init__(self, n_embd=16):
        super().__init__(n_embd)

    def forward(self, x_cur, x_prev):
        # temporal convolution
        x_cur, x_prev = self.convolution(x_cur, x_prev)

        # attention with residual connection: query = x_prev, key = value = x_cur
        x = x_cur + self.attention(x_prev, x_cur, x_cur)

        # layer normalisation over embedding dimension
        x = self.ln(x)

        # fully connected layers & dropout
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        return x














