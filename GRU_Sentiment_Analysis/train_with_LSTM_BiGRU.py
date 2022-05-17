# encoding=utf-8

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from dataset import IMDB
from model import LSTM_BiGRU


def train(epochs, use_gpu, optimizer, criterion, net, train_loader, validation_loader):
    train_acc, validation_acc = [], []
    early_stop, best_loss = 5, 100
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()
        for data in train_loader:
            inputs, train_labels = data

            if use_gpu:
                inputs, labels = inputs.cuda(), train_labels.cuda()
            
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, dim=1)
            train_correct += (train_predicted == labels).sum()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train: %d, epoch loss: %.3f,  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        train_acc.append(float(train_correct / train_total))

        # 模型测试
        correct = 0
        validation_loss = 0.0
        validation_total = 0
        net.eval()
        for data in validation_loader:
            input, labels = data
            if use_gpu:
                input, labels = input.cuda(), labels.cuda()
            
            outputs = net(input)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            validation_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        acc = float(correct / validation_total)
        avg_loss = validation_loss / validation_total
        validation_acc.append(acc)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop = 5
            torch.save(net, './models/LSTM_GRU.pth')
        else:
            early_stop -= 1

        if (epoch + 1) % 10 == 0 or early_stop == 0:
            print('')
            print('validation on validation set: %d, epoch loss: %.3f,  acc: %.3f \n' %
                  (epoch + 1, validation_loss / validation_total, 100 * correct / validation_total))
            if early_stop == 0:
                break

    return train_acc, validation_acc


def main():
    training_raw = IMDB(mode='train')

    # 8-2划分训练集和验证集
    train_size = int(0.8 * len(training_raw))
    validation_size = len(training_raw) - train_size
    training_set, validation_set = random_split(training_raw, [train_size, validation_size])

    # 设置超参数
    epochs = 100
    batch_size = 50
    use_gpu = torch.cuda.is_available()

    # 得到对应dataloader
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    network = LSTM_BiGRU()
    if use_gpu:
        network = network.cuda()
    print(network)

    # 定义loss和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=0.0003, weight_decay=5e-4)
    train_acc, validation_acc = train(epochs, use_gpu, optimizer, criterion, network, train_loader, validation_loader)
    # print(train_acc, validation_acc)
    x = [i + 1 for i in range(len(train_acc))]
    plt.plot(x, train_acc, 'r', label='Training acc')
    plt.plot(x, validation_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    main()
