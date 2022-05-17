# encoding=utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import IMDB


def evaluate(use_gpu, criterion, net, test_loader):
    # 模型测试
    correct = 0
    validation_loss = 0.0
    validation_total = 0
    net.eval()

    for data in test_loader:
        input, labels = data
        if use_gpu:
            input, labels = input.cuda(), labels.cuda()
        
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()
        validation_total += labels.size(0)
        correct += (predicted == labels.data).sum()
    
    avg_loss = validation_loss / validation_total
    acc = float(correct / validation_total)

    return avg_loss, acc


def main():
    print("Loading Dateset...")
    test_dataset = IMDB(mode='test')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=50)
    
    use_gpu = torch.cuda.is_available()
    criterion = nn.NLLLoss()

    print("Instantiating models...")
    net = torch.load('./models/HAN_5.pth')
    
    avg_loss, test_acc = evaluate(use_gpu, criterion, net, test_dataloader)
    print(avg_loss, test_acc)


if __name__ == "__main__":
    main()
