'''
Adaptive linear filter
Takes in accelerometer data as input into a CNN, where the adaptive linear filter is the loss function
'''

import numpy as np
import torch
import torch.nn as nn

class AdaptiveLinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2D expects (batch_size, in_channels, height, width)

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 21), padding='same')
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 1), padding='valid')

    def forward(self, X):
        '''
        Define forward pass of adaptive filter model
        :param X: shape (batch_size, 1, 3, 256)
        :return: X: shape (batch_size, 256)
        '''

        X = self.conv1(X)                   # 1st conv layer
        # no specified activation function (linear)
        X = self.conv2(X)              # 2nd conv layer

        # remove redundant dimensions
        return torch.squeeze(X)

    def adaptive_loss(self, y_true, y_pred):
        '''
        Apply adaptive filter
        custom loss function: MSE( FFT(CNN output) , FFT(raw PPG signal) )
        :param y_true: raw PPG signal
        :param y_pred: predicted motion artifacts from CNN
        :return: mean squared error between y_true and y_pred
        '''

        # take FFT
        y_true_fft = torch.fft.fft(y_true)
        y_pred_fft = torch.fft.fft(y_pred)

        # calculate MSE where error = raw ppg - motion artifact estimate
        e = torch.abs(y_true_fft - y_pred_fft) ** 2

        return torch.mean(e)


