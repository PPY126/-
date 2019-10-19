import numpy as np
import os
import jieba
import re

#加载数据集
def loadDataSet(pos,neg):
    '''
    :param pos:多少条非低素质类弹幕
    :param neg:多少条低素质类弹幕
    '''
    trainingList = []  #训练集
    classVec = []   #分类向量

    #录入非低素质类弹幕相关的训练集
    posList = list(open(pos,encoding = 'utf-8').readlines())
    posVec = [0] * len(posList)
    trainingList += posList
    classVec += posVec

    #录入低素质类弹幕相关的训练集
    negList = list(open(neg,encoding = 'utf-8').readlines())
    negVec = [1] * len(negList)
    trainingList += negList
    classVec += negVec
    
    return trainingList,classVec


#创建词汇表
def createVocabList(dataSet):
    vocabSet = set() #集合的元素唯一
    for doc in dataSet: 
        vocabSet = vocabSet|set(doc)
        vocabList = list(vocabSet)
    return vocabList #得到词汇表


#生成词向量函数
def setOfWords2Vec(vocabList,inputSet):
    
    returnVec = [0] * len(vocabList)
    for word in inputSet:  #inputSet是dataSet的每一条列表,inputSet元素数量小于词汇表vocabList
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


#创建词条向量列表
def get_trainMat(dataSet):
    trainMat = []  #词条向量列表
    vocabList = createVocabList(dataSet)
    for inputSet in dataSet:
        returnVec = setOfWords2Vec(vocabList,inputSet)
        trainMat.append(returnVec)
    return trainMat

def clean_str(string):
    """
    1. 将除汉字外的字符转为一个空格
    2. 除去句子前后的空格字符
    """
    string = re.sub(r'[^\u4e00-\u9fff]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()


#对训练集分词后返回词汇表和词条向量表
def jieba_cut(dataSet):
    lines = [list(jieba.cut(clean_str(line),cut_all=False)) for line in dataSet]
    lines = [[word for word in line if word != ' '] for line in lines]
    vocabulary = createVocabList(lines)
    trainMat = get_trainMat(lines)
    return vocabulary,trainMat


#对测试集进行分词
def jieba_cut1(dataSet):
    lines = [list(jieba.cut(line,cut_all=False)) for line in dataSet]
    lines = [[word for word in line if word != ' '] for line in lines]
    return lines


if __name__ == '__main__':
    print('')
