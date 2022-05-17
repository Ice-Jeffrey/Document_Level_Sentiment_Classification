# encoding=utf-8

from unicodedata import bidirectional
import torch
from torch import nn


class CNN_BiGRU(nn.Module):
    def __init__(self):
        super(CNN_BiGRU, self).__init__()

        # 定义基本的网络结构
        # nn.Conv2d()
        self.conv1 = nn.Conv1d(200, 50, 1)
        self.conv2 = nn.Conv1d(200, 50, 2)
        self.conv3 = nn.Conv1d(200, 50, 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(50, hidden_size=16, bidirectional=True)
        self.linear_layer = nn.Linear(32, 2)

    def CNN(self, input):
        # input: batch * num_of_sentences * length * 200
        batch_size = input.shape[0]
        input = input.view(-1, input.shape[-2], input.shape[-1])
        input = input.transpose(-1, -2) # (batch * num_of_sentences) * 200 * length

        output1 = self.conv1(input)    # (batch * num_of_sentences) * 50 * length
        output1 = self.pool(output1)    # (batch * num_of_sentences) * 50 * 1
        output1 = self.tanh(output1)

        output2 = self.conv2(input)
        output2 = self.pool(output2)
        output2 = self.tanh(output2)

        output3 = self.conv3(input)    # batch * num_of_sentences * length * 50
        output3 = self.pool(output3)
        output3 = self.tanh(output3)

        cnn_output = (output1 + output2 + output3) / 3
        cnn_output = torch.squeeze(cnn_output)  # (batch * num_of_sentences) * 50
        cnn_output = cnn_output.view(batch_size, -1, cnn_output.shape[-1])
        return cnn_output

    def GRU(self, input):
        # input: batch * num_of_sentences * 50
        input = input.transpose(0, 1)   # num_of_sentences * batch * 50
        output, _ = self.gru(input) # num_of_sentences * batch * (2 * hidden_size)
        output = torch.mean(output, dim=0)  # batch * (2 * hidden_size)
        return output

    def forward(self, x):
        cnn_output = self.CNN(x)
        output = self.GRU(cnn_output)    # batch * (hidden_size * 2)
        output = self.linear_layer(output)

        return output


class LSTM_BiGRU(nn.Module):
    def __init__(self):
        super(LSTM_BiGRU, self).__init__()

        # 定义基本的网络结构
        self.lstm = nn.LSTM(200, hidden_size=25, bidirectional=True)
        self.gru = nn.GRU(50, hidden_size=16, bidirectional=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax()
        )

    def word_encoder(self, input):
        # input: batch * num_of_sentences * length * 200
        batch_size = input.shape[0]
        input = input.transpose(1, 2).transpose(0, 1)   # length * batch * num_of_sentences * 200
        input = input.view(input.shape[0], -1, input.shape[-1])   #  length * (batch * num_of_sentences) * 200
        output, _ = self.lstm(input)
        output = torch.mean(output, dim=0)   # (batch * num_of_sentences) * 50
        output = output.view(batch_size, -1, output.shape[-1]).transpose(0, 1)  # num_of_sentences * batch * 50
        return output

    def sentence_encoder(self, input):
        # input: num_of_sentences * batch * 50
        output, _ = self.gru(input) # num_of_sentences * batch * (2 * hidden_size)
        output = torch.mean(output, dim=0)  # batch * (2 * hidden_size)
        return output

    def forward(self, x):
        sentence_vector = self.word_encoder(x)
        document_vector = self.sentence_encoder(sentence_vector)    # batch * (hidden_size * 2)
        output = self.linear_layer(document_vector)

        return output


# def main():
#     print("Hello PyTorch!")


# if __name__ == "__main__":
#     main()
