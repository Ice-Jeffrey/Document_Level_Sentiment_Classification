# encoding=utf-8

import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
# setup_seed(20)


class Self_Attention(nn.Module):
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(Self_Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, key, value):
        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (sequence_length, batch_size, hidden_dim * 2)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        key = key.transpose(0, 1).transpose(1, 2) # (batch_size, hidden_dim * 2, sequence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sequence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sequence_length's dimension

        value = value.transpose(0, 1) # (batch_size, sequence_length, hidden_dim * 2)
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim * 2)
        attention_output = attention_output.squeeze(1) # (batch_size, hidden_dim * 2)

        return attention_output, attention_weight.squeeze(1)


class HAN(nn.Module):
    def __init__(self):
        super(HAN, self).__init__()

        # 定义基本的网络模块
        self.batch_size = 0
        self.num_of_sentences = 0
        self.word_gru = nn.GRU(200, hidden_size=50, bidirectional=True)
        self.sentence_gru = nn.GRU(100, hidden_size=50, bidirectional=True)
        self.wattention = Self_Attention(100)
        self.sattention = Self_Attention(100)

        self.u_w = torch.randn(100)
        self.word_MLP = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh()
        )

        self.u_s = torch.randn(100)
        self.sentence_MLP = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh()
        )

        self.linear_layer = nn.Linear(100, 2)

        if torch.cuda.is_available():
            self.wattention = self.wattention.cuda()
            self.sattention = self.sattention.cuda()
            self.u_w = self.u_w.cuda()
            self.u_s = self.u_s.cuda()

    def word_encoder(self, input):
        # input: batch * num_of_sentences * length * 200
        self.batch_size = input.shape[0]
        self.num_of_sentences = input.shape[1]
        input = input.transpose(1, 2).transpose(0, 1)   # length * batch * num_of_sentences * 200
        input = input.view(input.shape[0], -1, input.shape[-1])   #  length * (batch * num_of_sentences) * 200
        output, _ = self.word_gru(input)    # length * (batch * num_of_sentences) * 100
        return output

    def word_attention(self, input):
        # input_shape: length * (batch * num_of_sentences) * 100
        query = torch.stack([self.u_w for i in range(self.batch_size * self.num_of_sentences)], dim=0)  # batch * 100    
        dim0, dim1 = input.shape[0], input.shape[1]
        temp_input = input.view(-1, 100)
        temp_key = self.word_MLP(temp_input)
        key = temp_key.view(dim0, dim1, 100)
        value = input
        attention_output, attention_weight = self.wattention(query, key, value)
        # attention_output: (batch * num_of_sentences) * 100
        attention_output = attention_output.view(self.batch_size, self.num_of_sentences, -1).transpose(0, 1)    # num_of_sentences * batch * hidden_size
        return attention_output

    def sentence_encoder(self, input):
        # input: num_of_sentences * batch * 100
        output, _ = self.sentence_gru(input) # num_of_sentences * batch * 100
        return output

    def sentence_attention(self, input):
        # input_shape: num_of_sentences * batch * 100
        query = torch.stack([self.u_s for i in range(self.batch_size)], dim=0)  # batch * 100    
        dim0, dim1 = input.shape[0], input.shape[1]
        temp_input = input.view(-1, 100)
        temp_key = self.sentence_MLP(temp_input)
        key = temp_key.view(dim0, dim1, 100)
        value = input
        attention_output, attention_weight = self.sattention(query, key, value)
        # attention_output: batch * 100
        return attention_output

    def forward(self, x):
        sentence_vector = self.word_encoder(x)
        sentence_vector = self.word_attention(sentence_vector)
        document_vector = self.sentence_encoder(sentence_vector)
        document_vector = self.sentence_attention(document_vector)
        output = self.linear_layer(document_vector)

        return output


# def main():
#     print("Hello PyTorch!")


# if __name__ == "__main__":
#     main()
