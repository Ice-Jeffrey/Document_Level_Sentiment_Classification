# encoding=utf-8
# 得到语料库中每个单词对应的词向量


import os
from genericpath import isdir
from gensim.models import Word2Vec


dir = 'F:/Sentiment_Analysis/dataset'


def getSentences():
    genre = ['train', 'test']
    sentences = []
    for g in genre:
        directory = os.path.join(dir, g)
        tags = ['neg', 'pos']
        for tag in tags:
            tempdir = os.path.join(directory, tag)
            files = os.listdir(tempdir)
            for file in files:
                print(os.path.join(tempdir, file))
                with open(os.path.join(tempdir, file), 'r', encoding='utf-8') as f:
                    lines = f.read().split('\n')
                    l = [line.split() for line in lines]
                    # print(l)
                    sentences += l

    return sentences


def main():
    sentences = getSentences()
    # print(sentences[0])
    # exit()

    model = Word2Vec(sentences, vector_size=200, min_count=0, sg=1)
    word_vectors = model.wv
    if not os.path.isdir('./word2vec'):
        os.mkdir('./word2vec')
    word_vectors.save('./word2vec/word_vectors.dat')
    print('Word vectors is ready!')

    # test
    # vector = model.wv['chantings']  # get numpy vector of a word
    # print(vector)
    # sims = model.wv.most_similar('chantings', topn=10)  # get other similar words
    # print(sims)


if __name__ == "__main__":
    main()
