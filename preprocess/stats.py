# encoding=utf-8
# 统计得到文章中句子长度及句子中单词长度的统计量


import os
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from matplotlib import pyplot as plt


dir = 'F:/Sentiment_Analysis/aclImdb'


def getStats():
    num_of_sentences = {}
    num_of_words = {}

    genre = ['train', 'test']
    for g in genre:
        directory = os.path.join(dir, g)
        tags = ['neg', 'pos']
        for tag in tags:
            tempdir = os.path.join(directory, tag)
            files = os.listdir(tempdir)
            for file in files:
                # print(os.path.join(tempdir, file))
                with open(os.path.join(tempdir, file), 'r', encoding='utf-8') as f:
                    lines = f.read().replace('<br /><br />', ' ')   # 去掉回车换行符号
                    lines = sent_tokenize(lines)    # 分句
                    lines = [line.lower().translate(str.maketrans('', '', string.punctuation)) for line in lines]   # 将句子中的单词变为小写并去掉标点
                    l = [word_tokenize(line) for line in lines] # 分词
                    
                    num_of_sentences[len(lines)] = num_of_sentences.get(len(lines), 0) + 1
                    for line in lines:
                        num_of_words[len(line)] = num_of_words.get(len(line), 0) + 1

    return num_of_sentences, num_of_words


def main():
    num_of_sentences, num_of_words = getStats()
    num_ranking_sentences = sorted(num_of_sentences.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:10]
    num_ranking_words = sorted(num_of_words.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:10]

    data = [elem[1] for elem in num_ranking_sentences]
    labels = [str(elem[0]) for elem in num_ranking_sentences]
    plt.bar(range(10), data, tick_label=labels)
    plt.show()

    data = [elem[1] for elem in num_ranking_words]
    labels = [str(elem[0]) for elem in num_ranking_words]
    plt.bar(range(10), data, tick_label=labels)
    plt.show()
    # print(num_of_words)
    # print(num_of_sentences)


if __name__ == "__main__":
    main()
