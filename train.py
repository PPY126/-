import bayes
from data import *
from sklearn.externals import joblib
import time

pos = "G:/Python/基于朴素贝叶斯算法的低素质类弹幕分类器/测试1/train_非低素质类弹幕.txt"
neg = "G:/Python/基于朴素贝叶斯算法的低素质类弹幕分类器/测试1/train_低素质类弹幕.txt"

start = time.clock()#训练模型开始时间

print("正在获取训练矩阵及其分类向量")
trainingList,classVec = loadDataSet(pos,neg)

print("正在将训练矩阵分词，并生成词表")
vocabList,trainMat = jieba_cut(trainingList) #创建词汇表

bayes = bayes.BerNB(vocabList)

print("正在训练模型")
bayes.trainNB(trainMat,classVec)

print("保存模型")
joblib.dump(bayes, "train_model.m")

print ("训练模型使用了：" + str(time.clock() - start) + '秒\n')
