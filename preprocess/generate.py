# encoding=utf-8
# 生成标准化之后的数据集


import os
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec


sentence_max = 7
word_max = 73
dir = 'F:/Sentiment_Analysis/aclImdb'
new_dir = 'F:/Sentiment_Analysis/dataset'


def normalize_line(s):
    index, flag = 0, 0
    if len(s) < word_max:
        index = len(s)
        flag = 1
    else:
        index = word_max

    sentence = s[:index]
    if flag == 1:
        for i in range(word_max - len(s)):
            sentence.append('<pad>')
    return sentence


def normalize_lines(l):
    index, flag = 0, 0
    if len(l) >= sentence_max:
        index = sentence_max
    else:
        index = len(l)
        flag = 1

    lines = []
    l = l[:index]
    for line in l:
        line = normalize_line(line)
        lines.append(line)
    
    if flag == 1:
        temp_list = []
        for i in range(word_max):
            temp_list.append('<pad>')
        for i in range(sentence_max - len(l)):
            lines.append(temp_list)

    return lines


def process():
    genre = ['train', 'test']
    sentences = []
    for g in genre:
        directory = os.path.join(dir, g)
        to_directory = os.path.join(new_dir, g)
        tags = ['neg', 'pos']
        for tag in tags:
            tempdir = os.path.join(directory, tag)
            to_dir = os.path.join(to_directory, tag)
            files = os.listdir(tempdir)
            for file in files:
                to_path = os.path.join(to_dir, file)
                with open(os.path.join(tempdir, file), 'r', encoding='utf-8') as f:
                    lines = f.read().replace('<br /><br />', ' ')   # 去掉回车换行符号
                    lines = sent_tokenize(lines)    # 分句
                    lines = [line.lower().translate(str.maketrans('', '', string.punctuation)) for line in lines]   # 将句子中的单词变为小写并去掉标点
                    l = [word_tokenize(line) for line in lines] # 分词，得到代表文章的矩阵，需要对该矩阵进行标准化处理

                    lines = normalize_lines(l)
                    print(to_path)
                    with open(to_path, 'w', encoding='utf-8') as write_file:
                        lines = [(' ').join(line) for line in lines]
                        write_file.write(('\n').join(lines))


def main():
    process()
    # sentences = getSentences()
    # # print(sentences[0])
    # # exit()

    # model = Word2Vec(sentences, vector_size=200, min_count=0, sg=1)
    # word_vectors = model.wv
    # word_vectors.save('./word2vec/word_vectors.dat')
    # print('Word vectors is ready!')

    # test
    # vector = model.wv['chantings']  # get numpy vector of a word
    # print(vector)
    # sims = model.wv.most_similar('chantings', topn=10)  # get other similar words
    # print(sims)


if __name__ == "__main__":
    main()
