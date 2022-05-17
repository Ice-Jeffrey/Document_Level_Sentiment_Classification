#encoding=utf-8
# 自定义Dataloader

import os
import torch
from torch.utils.data import Dataset
from gensim.models import KeyedVectors


class IMDB(Dataset):
    def __init__(self, mode):
        self.count = 0
        self.mode = mode
        self.wv = KeyedVectors.load("./word2vec/word_vectors.dat", mmap='r')    # 载入训练好的word2vec词向量
        # print(self.wv)

        path = 'F:/Sentiment_Analysis/dataset'
        if mode != 'train' and mode != 'test':
            print('No dataset loaded!')
        self.path = os.path.join(path, mode)

        self.data = []
        self.tags = []
        labels = ['neg', 'pos']
        for i, label in enumerate(labels):
            tempdir = os.path.join(self.path, label)
            self.count += len(os.listdir(tempdir))
            for file in os.listdir(tempdir):
                self.tags.append(i)
                with open(os.path.join(tempdir, file), 'r', encoding='utf-8') as f:
                    lines = f.read().split('\n')
                    matrix = [line.split() for line in lines]
                    
                    # 将词语映射为词向量
                    vectors = []
                    for sentence in matrix:
                        temp_vector = []
                        for word in sentence:
                            temp = self.wv[word]
                            temp = torch.from_numpy(temp)
                            temp_vector.append(temp)
                        
                        if len(temp_vector) > 0:
                            vectors.append(torch.stack(temp_vector))    # 将句子变为一个二维tensor
                    self.data.append(torch.stack(vectors))  # 将文章变为一个三维tensor
    
    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.data[index], self.tags[index]


# if __name__ == "__main__":
#     temp = IMDB('train')
#     print(temp[0])
