import numpy as np
from data import setOfWords2Vec

class BerNB:
    
    def __init__(self,vocabulary):
        self.p1Vect = None
        self.p0Vect = None
        self.p1 = None
        self.vocabulary = vocabulary

    def trainNB(self,trainMat,classVec):
        n = len(trainMat)  #训练的文档数目
        m = len(trainMat[0])  #每篇文档的词条数
        pAb = sum(classVec)/n #文档属于侮辱类的概率
        p0Num = np.ones(m)   #词条出现数初始化为1
        p1Num = np.ones(m)
        p0Denom = 2.0
        p1Denom = 2.0 #分母初始化为2
        for i in range(n):
            if classVec[i] == 1:
                p1Num += trainMat[i]        #全部特征是否在A类出现
                p1Denom += sum(trainMat[i])  #属于A类的有多少个特征
            else:
                p0Num += trainMat[i]
                p0Denom += sum(trainMat[i])
        p1V = np.log(p1Num/p1Denom)
        p0V = np.log(p0Num/p0Denom)

        self.p0Vect = p0V
        self.p1Vect = p1V
        self.p1 = pAb

    def classify_danmu(self, danmu):
        """
        分类函数,对输入弹幕进行处理，然后分类
        :param vec2Classify: 欲分类的新闻
        """
        testMat = setOfWords2Vec(self.vocabulary,danmu)
        return self.classifyNB(testMat)

    #分类函数
    def classifyNB(self,vec2Classify):
        p1 = sum(vec2Classify * self.p1Vect) + np.log(self.p1)
        p0 = sum(vec2Classify * self.p0Vect) + np.log(1-self.p1)
        if p1>p0:
            return 1
        else:
            return 0
