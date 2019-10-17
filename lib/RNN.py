"""
Create the Recurrent Neural Network (RNN) for PyTorch
"""
import torch
import numpy as nn

from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0):
        """
        Initialize the RNN

        :param input_size:      The size of the input
        :param output_size:     The size of the output
        """
        # Call the super class
        super(RNN, self).__init__()

        # Set the RNN layer
        self.rnn = nn.RNNCell(input_size, output_size)
        self.dropout_prob = dropout_prob

        # Random state
        self.hx = torch.nn.Parameter(torch.randn(output_size), requires_grad=True)

    def forward(self, X):
        """
        Do the forward propagation of the data

        :param X:   The input
        :return:    The outputs of the RNN
        """
        iterations = X.size()[0]

        # Do the iterations
        outputs = []

        # Adjust size of hx
        hx = self.hx.repeat(X[0].size()[0]).view(X[0].size()[0], -1)
        for iteration in range(iterations):
            # Add dropout
            dropout = nn.Dropout(self.dropout_prob)

            # Do a step of the forward
            input = X[iteration]
            hx = self.rnn(dropout(input), hx)

            # Append the result
            outputs.append(hx)

        return outputs

