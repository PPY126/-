import bayes
from data import *
from sklearn.externals import joblib
import time

pos = "G:/Python/基于朴素贝叶斯算法的低素质类弹幕分类器/测试1/test_非低素质类弹幕.txt"
neg = "G:/Python/基于朴素贝叶斯算法的低素质类弹幕分类器/测试1/test_低素质类弹幕.txt"

start = time.clock()#测试模型开始时间

print("正在得到测试矩阵及其分类向量")
testingList,classVec = loadDataSet(pos,neg)

nb = joblib.load("train_model.m")
# 读取模型

testlines= jieba_cut1(testingList) #测试样本向量化


resultVec = []  #测试结果向量集
for testline in testlines:
    if nb.classify_danmu(testline)==1:
        resultVec += [1]
    else:
        resultVec += [0]


correct = 0  #初始化分类正确的弹幕数
recall1 = 0
accuracy = 0

for i in range(len(classVec)):
    
    #print(testingList[i],classVec[i],resultVec[i],'\n')

    #测试结果与原测试集分类标签对比，相等则分类正确弹幕数+1
    if resultVec[i] == classVec[i]:
        correct += 1
    if resultVec[i]==classVec[i]==1:
        recall1 += 1
    if classVec[i]==1:
        accuracy+=1

acc = correct/len(classVec)  #正确率为
Recall1 = recall1/sum(resultVec)
Accuracy = recall1/accuracy
print("正确率为："+str(acc))
print("召回率为："+str(Recall1))
print("精确率为："+str(Accuracy))
print ("测试模型使用了：" + str(time.clock() - start) + '秒\n')
