from torch.nn import Module
import torch
from torch.nn import ModuleList

import math
import torch.nn.functional as F

from models.Encoder import Encoder


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_hz: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])
        self.encoder_list_3 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])
        #         step,cha,hz

        self.embedding_input = torch.nn.Linear(d_channel * d_hz, d_model)
        self.embedding_channel = torch.nn.Linear(d_input * d_hz, d_model)
        self.embedding_hz = torch.nn.Linear(d_input * d_channel, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_hz, 3)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_hz, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        forward propagation
        :param x: enter
        :param stage: Used to describe whether it is the training process of the training set or the testing process of the test set at this time  The mask mechanism is not added during the test
        :return: Output, two-dimensional vector after gate, step-wise The score matrix in the encoder, the score matrix in the channel-wise encoder, the three-dimensional matrix after step-wise embedding, the three-dimensional matrix after channel-wise embedding, gate
        """
        # step-wise
        # The score matrix is input， Add mask by default and pe
        step_x = x
        step_x = step_x.reshape(step_x.shape[0], step_x.shape[1], step_x.shape[2] * step_x.shape[3])
        encoding_1 = self.embedding_input(step_x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe
        # Sample Time Channel Hertz
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # The score matrix is channel without mask and pe by default
        channel_x = x
        channel_x = channel_x.transpose(2, 1)
        channel_x = channel_x.reshape(channel_x.shape[0], channel_x.shape[1], channel_x.shape[2] * channel_x.shape[3])
        encoding_2 = self.embedding_channel(channel_x)
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # hz-wise
        hz_x = x
        hz_x = hz_x.transpose(3, 1)
        hz_x = hz_x.reshape(hz_x.shape[0], hz_x.shape[1], hz_x.shape[2] * hz_x.shape[3])
        encoding_3 = self.embedding_hz(hz_x)
        channel_to_gather = encoding_3

        for encoder in self.encoder_list_3:
            encoding_3, score_channel = encoder(encoding_3, stage)

        # 3D to 2D
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)
        encoding_3 = encoding_3.reshape(encoding_3.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2, encoding_3], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2], encoding_3 * gate[:, 2:3]], dim=-1)

        # output
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
