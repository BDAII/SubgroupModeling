import os
import sys
import time
import random
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from util import *
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
warnings.filterwarnings(action='ignore')

class TreeNode:
    def __init__(self, models, trainSampleIndex, validSampleIndex, testSampleIndex, auc, depth, 
            parentIndex, parentSplitFeature, parentFeatureValue, featurePath ):
        self.auc = auc                   # 当前节点的AUC
        self.models = models             # 对当前结点建模的模型    
        self.curDepth = depth            # 当前节点所处深�?
        self.trainSampleIndex = trainSampleIndex
        self.validSampleIndex = validSampleIndex
        self.testSampleIndex = testSampleIndex
        self.featurePath = featurePath   # 从根节点到达当前节点的全部（划分）特�?
        self.parent = parentIndex        # 父节点下�?
        self.parentSplitFeature = parentSplitFeature # 父节点分割时所用的特征�?
        self.parentFeatureValue = parentFeatureValue # 父节点分裂得到当前节点时，当前节点特征的取�?
        
        # 预分割选择父节点作为划分后需要设置的属性（调用setInnerNone/setLeafNode�?
        self.akiRate = None              # 当前节点的AKI�?
        self.index = -1                  # 当前节点的序�?
        self.isLeaf = True               # 是否为叶子节点，默认为True
        self.splitFeature = None         # 分裂当前节点所用特征的下标   （内部节点有效）
        self.splitValue = None           # 分裂当前节点的特征取�?     （内部节点有效）
        self.subNodes = []               # 当前节点的子节点           （内部节点有效）
    
    def trainSample(self, sample):
        return sample.loc[self.trainSampleIndex]
    
    def validSample(self, sample):
        return sample.loc[self.validSampleIndex]
    
    def testSample(self, sample):
        return sample.loc[self.testSampleIndex]

    def setInnerNode(self, splitFeature, splitValue, index):
        self.isLeaf = False
        self.splitFeature = splitFeature
        self.splitValue = splitValue
        self.index = index
    
    def setLeafNode(self, index):
        self.index = index
        self.isLeaf = True
        self.splitFeature = None
        self.splitValue = None
        self.subNodes = None
     
class DecisionTree:
    def __init__(self, minSamplesLeaf=200, minSampleSplit=400, maxDepth=4, fixedFeature=[], logger=None):
        self.minSamplesLeaf = minSamplesLeaf
        self.minSampleSplit = minSampleSplit
        self.maxDepth = maxDepth
        self.fixedFeature = fixedFeature
        self.rawSamples = []
        self.trainSample = []
        self.validSample = []
        self.rawTestSamples = []
        self.baseAuc = 0.0
        self.__counter = 0
        self.overallAuc = 0.0
        self.logger = logger
        self.leafNodeInfoList = []
        self.innerNodeInfoList = []
        self.maxSizeToTransfer = 5000 # 如果子节点样本大于这个值，则不迁移
        self.baseModelFeature = []
        self.nodeMap = {}
        # self.predict = [np.zeros(0) for i in range(10)]
        # self.basePredict = [np.zeros(0) for i in range(10)]
        self.predict = np.zeros(0)
        self.basePredict = np.zeros(0)
        self.Y = np.zeros(0)
        self.savedFeatureSizeTable = []
        self.subgroupFeatureRands = pd.DataFrame(["null" for i in range(801)], 
                                                index = [i for i in range(1, 802)])
  
    def fitThenPredict(self, samples, testSamples):
        self.rawSamples, self.rawTestSamples = samples, testSamples
        self.trainSample, self.validSample = trainTestSamplesSplit(samples, 0.2, True)
        self.kBestFeature, self.featurePValues = getKBestFeatureSample(self.rawSamples, 
                                                                    len(self.rawSamples.columns) - 1)
        # compute then save root node's feature rank and p-value
        self.savedFeatureSizeTable = [10, 20, 50, 100, 250, 500, 1000, 2000, len(self.rawSamples.columns)]
        aCol = [0, False]
        aCol.extend([-1 for i in range(len(self.savedFeatureSizeTable))])
        aCol.extend(self.kBestFeature)
        self.importFeatureTable = pd.DataFrame(aCol).copy(deep=True)
        aCol = [0, False]
        aCol.extend(self.featurePValues)
        self.featurePValueTable = pd.DataFrame(aCol).copy(deep=True)

        # build a tree node then begin split
        self.rootNodeModels = trainLightgbmCV(self.rawSamples.values, 10)
        root = TreeNode(self.rootNodeModels, self.trainSample.index, self.validSample.index, 
            self.rawTestSamples.index, self.baseAuc, depth=1, parentIndex=0, parentSplitFeature="--", 
            parentFeatureValue="--", featurePath=[])
        root.dValue = 999
        self.tree = self.__createTree(root, 0)

        # print log
        self.logger.info("".center(80, "="))
        for msg in self.leafNodeInfoList:
            self.logger.info(msg)
        for msg in self.innerNodeInfoList:
            self.logger.info(msg)

        # print some metric
        subAuc = round(roc_auc_score(self.Y, self.predict), 4)
        baseAuc = round(roc_auc_score(self.Y, self.basePredict), 4) 
        dValue = round(subAuc - baseAuc)

        subF1score, baseF1score, f1DValue, subRecall, baseRecall, recallDValue, subPrecision, basePrecision, precDValue \
            = self.__showMetric(self.Y, self.predict, self.basePredict) 

        self.logger.info("subAuc {}, baseAuc {}, dvalue {}".format(subAuc, baseAuc, dValue))
        self.logger.info("subF1Score {}, baseF1Score {}, dvalue {}".format(subF1score, baseF1score, f1DValue))
        self.logger.info("subRecall {}, baseRecall {}, dvalue {}".format(subRecall, baseRecall, recallDValue))
        self.logger.info("subPrecision {}, basePrecision {}, dvalue {}".format(subPrecision, basePrecision, precDValue))

        # save subgroup's top k feature.
        dirName = str(int(time.time()))
        self.logger.info("dir name {}".format(dirName))
        os.makedirs(os.path.join(Constant.trainDataDir, dirName))
        featureTableFileName = os.path.join(Constant.trainDataDir, dirName, "importantFeatureTable.csv")
        featurePValueFileName = os.path.join(Constant.trainDataDir, dirName, "featurePValueTable.csv")
        self.featurePValueTable.to_csv(featurePValueFileName)
        self.importFeatureTable.to_csv(featureTableFileName)
        self.logger.info("data saved!")
        return self

    def __createTree(self, node, parentIndex):
        node.backNodeIndex = -1
        node.parent = parentIndex
        allPreSplitResults =  self.__getAllPreSplit(node)

        # build model after merge train sample and valid sample
        node.oldModels = node.models  
        node.trainSampleIndex = pd.concate((node.trainSampleIndex, node.validSampleIndex))
        if parentIndex != 0:
            node.selfModels = trainLightgbmCV(node.trainSample(self.rawSamples).values, 10) 
        else:
            node.selfModels = node.models
        node.models = node.selfModels

        # if dvalue lower than 0, roll back to grandparent's model(find a best model) 
        node.upm = False
        if node.dValue < 0.005 and node.parent > 1: 
            parentSize, parentDown = 0, 0
            grandParent = self.nodeMap[node.parent]
            nodeValidSample = node.validSample(self.rawSamples)
            while grandParent.index > 1:
                parentSize += 1
                newSubAuc = np.mean(multilModelPredictAUC(grandParent.selfModels, nodeValidSample))
                newDValue = round(newSubAuc - baseAuc, 4)
                if newDValue > node.dValue:
                    node.auc = newSubAuc
                    node.dValue = newDValue
                    node.models = grandParent.selfModels # node.model may reference from parent's model, so use selfModels
                    node.backNodeIndex = grandParent.index
                    node.upm = True
                if newDValue < 0:
                    parentDown += 1
                if grandParent.parent in self.nodeMap.keys():
                    grandParent = self.nodeMap[grandParent.parent]
                else: break
            # if the auc of grandparent's model lower than 0, roll back to root node        
            if node.dValue < 0:  
                node.models = self.rootNodeModels
                node.backNodeIndex = 1
                node.upm = True

        # mark node a leaf node if not meet split condition
        if len(allPreSplitResults) == 0:
            node.setLeafNode(self.__getIndex())
            self.nodeMap[node.index] = node   # save node
            self.__showNodeInfo(node, True)
        else:
            bestSplit = self.__selectSplitByAuc(allPreSplitResults)
            splitFeature, splitValue = bestSplit["splitFeature"], bestSplit["splitValue"]
            node.subNodes = bestSplit["subNodes"]
            node.setInnerNode(splitFeature, splitValue, self.__getIndex())
            self.nodeMap[node.index] = node 

            # continue splitting
            for subNode in node.subNodes:
                subNode = self.__createTree(subNode, node.models, node.index)
            self.__showNodeInfo(node, False)
        return node

    def __getAllPreSplit(self, node):
        # return empty list if not meet split condition
        curNodeTrainSample = node.trainSample(self.rawSamples)
        curNodeValidSample = = node.validSample(self.rawSamples)
        curNodeTestSample = node.testSample(self.rawTestSamples)
        if not self.__splitCondition(node):
            return []

        # traversing all immutable features, find a best feature which has highest auc d-value 
        targetFeatureNames = list(set(self.fixedFeature) - set(node.featurePath))
        targetFeatureNames.sort()
        logs, maxOverallAuc, bestPreSplitResult = [], -9999.99, None 
        for splitFeature in targetFeatureNames:
            splitValue = self.__selectSplitValue(node, splitFeature)
            trainSubNodeMap = self.__splitNode(curNodeTrainSample, splitFeature, splitValue)
            validSubNodeMap = self.__splitNode(curNodeValidSample, splitFeature, splitValue)
            testSubNodeMap = self.__splitNode(curNodeTestSample, splitFeature, splitValue)

            # judge whether the subgroups are too small,and reject them if they are too small
            skipFlag = False
            for subNodeName, subNodeTrainSample in trainSubNodeMap.items():
                curNodeTrainSize = len(curNodeTrainSample)
                if len(subNodeTrainSample) > curNodeTrainSize - self.minSamplesLeaf:
                    skipFlag = True
                    break
            if skipFlag: continue

            # 对分割后的两个子节点进行建模
            nodeTrainSize = len(curNodeTrainSample)
            subNodes, leafDValueList, innerDValueList, cv = [], [], [], 10
            for subNodeName, subNodeTrainSample in trainSubNodeMap.items():
                newFeaturePath = node.featurePath.copy()
                newFeaturePath.append(splitFeature)
                subNodeValidSample = validSubNodeMap[subNodeName]
                subNodeTestSample = testSubNodeMap[subNodeName]
                size = len(subNodeTrainSample)

                # 为了加快速度，我们只对小亚组进行建模
                if (len(subNodeTrainSample) / nodeTrainSize) > 0.65: 
                    aSubNode = TreeNode([], subNodeTrainSample.index, subNodeValidSample.index, 
                        subNodeTestSample.index, -999, node.curDepth + 1, node.index, splitFeature, 
                        splitValue, newFeaturePath)
                    subNodes.append(aSubNode)
                    continue

                subModels = trainLightgbmCV(subNodeTrainSample.values, cv)
                subAucList = multilModelPredictAUC(subModels, subNodeValidSample.values)
                subAuc  = round(np.mean(subAucList), 4)

                baseAuc = multilModelPredictAUC(self.rootNodeModels, subNodeValidSample.values)
                aSubNode = TreeNode(subModels, subNodeTrainSample.index, subNodeValidSample.index, 
                                    subNodeTestSample.index, subAuc, node.curDepth + 1, node.index, 
                                    splitFeature, splitValue, newFeaturePath)
                dValue = round(subAuc-baseAuc, 4)
                aSubNode.dValue = dValue
                subNodes.append(aSubNode)
                leafDValueList.append(dValue)

                # ==================================================================================
                modelTestData = subNodeTestSample.values
                subTestAucs = multilModelPredictAUC(subModels, modelTestData)
                baseTestAucs = multilModelPredictAUC(self.rootNodeModels, modelTestData)
                s = round(np.sum(subTestAucs) / len(subTestAucs), 4)
                b = round(np.sum(baseTestAucs) / len(baseTestAucs), 4)
                d = round(s - b, 4)
                subF1score, baseF1score, f1DValue, subRecall, baseRecall, recallDValue, subPrecision, basePrecision, precDValue \
                    = self.__showMetric(testY, predict, basePredict) 
                # ==================================================================================

                logs.append("[{}] - value {}, size({},{}), auc({},{},{}) -- test:auc({},{},{}), f1-score({},{},{}), recall({},{},{}), prec({},{},{})"
                    .format(splitFeature.ljust(8, ' '), splitValue, len(subNodeTrainSample),
                    len(subNodeValidSample), subAuc, baseAuc, dValue, s, b, d,
                    subF1score, baseF1score, f1DValue, 
                    subRecall, baseRecall, recallDValue, 
                    subPrecision, basePrecision, precDValue))    
            
            # 计算auc差值，构造本次预分割结果
            overallAuc = np.max(leafDValueList)
            logs.append("-".center(80, "-"))
            aSplitResult = {
                "auc": overallAuc,
                "splitFeature": splitFeature,
                "splitValue": splitValue,
                "subNodes": subNodes,
                "logs": logs
            }
            
            # 从多个特征中选择出auc差值最大的那一次预分割
            if overallAuc > maxOverallAuc:
                maxOverallAuc = overallAuc
                bestPreSplitResult = aSplitResult

        # 补上大亚组的模型
        subNodes = bestPreSplitResult["subNodes"]
        for subNode in subNodes:
            if subNode.auc == -999:
                trainSample = subNode.trainSample(self.rawSamples)
                validSample = subNode.validSample(self.rawSamples)

                subModels = trainLightgbmCV(trainSample.values, cv)
                subAucList = multilModelPredictAUC(subModels, validSample.values)
                subAuc  = round(np.mean(subAucList), 4)
                baseAuc = round(np.mean(multilModelPredictAUC(self.rootNodeModels, validSample.values)), 4)

                subNode.auc = subAuc
                subNode.models = subModels
                subNode.dValue = round(subAuc - baseAuc, 4)
        # print log
        for log in logs:
            self.logger.info(log)
            
        return [bestPreSplitResult]

    def __splitCondition(self, node):
        samples = node.trainSample(self.rawSamples)
        notPureNode, gtMinSampleSplit, ltMaxDepth = False, False, False
        # 节点属于同一�?
        if len(set(samples["Label"])) > 1:
            notPureNode = True
        # 节点数量小于minSampleSplit
        if len(samples) >= self.minSampleSplit:
            gtMinSampleSplit = True
        # 当前节点处于maxDepth
        if node.curDepth < self.maxDepth:
            ltMaxDepth = True
        return notPureNode and gtMinSampleSplit and ltMaxDepth

    def __selectSplitValue(self, node, featureName):
        samples, targetValue = node.trainSample(self.rawSamples), None
        featureValues = samples[featureName].values
        uniqueFeatureValues = list(set(featureValues))

        # 遍历每个值，取划分后信息增益最大的那个值（参考《机器学习�?决策树章节）
        Y, T = samples["Label"].values, []
        sortedFeatureValues = np.sort(uniqueFeatureValues)
        D, entY, maxGain, targetValue = len(Y), self.__entropy(Y), -999999, 0
        for i in range(1, len(sortedFeatureValues)):
            t = float(sortedFeatureValues[i-1] + sortedFeatureValues[i]) / 2.0
            Dtp, Dtn = Y[featureValues>=t], Y[featureValues<t]
            if len(Dtp)==0 or len(Dtn)==0:
                continue
            gain = entY - (len(Dtp)/D*self.__entropy(Dtp) + len(Dtn)/D*self.__entropy(Dtn))
            if gain > maxGain:
                maxGain, targetValue = gain, t
        return targetValue

    def __splitNode(self, samples, splitFeature, splitValue):
        subNodeInfos = {}  # 字典格式:nodeName（splitValue�?nodeSampleIndexs
        if splitFeature == "DEMO_Age":
            # 第一�?
            childSample = samples.loc[samples["DEMO_Age"] <= 12]
            # 第二�?
            restTrain = samples.loc[samples["DEMO_Age"] > 12]
            normalSample = restTrain.loc[restTrain["DEMO_Age"] <= 65]
            # 第三�?
            oldSample = restTrain.loc[restTrain["DEMO_Age"] > 65]
            if len(childSample) < self.minSamplesLeaf:
                normalSample = pd.concat((childSample, normalSample), axis=0)
            else:
                subNodeInfos["child"] = childSample
            subNodeInfos["normal"] = normalSample
            subNodeInfos["old"] = oldSample
        else:
            subNodeInfos['gt'], subNodeInfos['lt'] = [], []
            for sampleIndex in samples.index:
                sample = samples.loc[sampleIndex]
                key = 'gt' if sample[splitFeature] > splitValue else 'lt'
                subNodeInfos[key].append(sampleIndex)
            subNodeInfos["gt"] = samples.loc[subNodeInfos["gt"]]
            subNodeInfos["lt"] = samples.loc[subNodeInfos["lt"]]
        return subNodeInfos

    def __getIndex(self):
        self.__counter = self.__counter + 1
        return self.__counter

    def __entropy(self, Y):
        size, aki = len(Y), np.sum(Y == 1)
        akiRate, noAkiRate = float(aki)/float(size), float(size-aki)/float(size)
        return -(akiRate*np.log2(akiRate) + noAkiRate*np.log2(noAkiRate))

    def __selectSplitByAuc(self, allPreSplitResults):
        bestSplit, maxAuc = None, -99999.0
        for aSplitResult in allPreSplitResults:
            if maxAuc < aSplitResult["auc"]:
                bestSplit, maxAuc = aSplitResult, aSplitResult["auc"]
        return bestSplit

    def __showNodeInfo(self, node, isLeaf=True):
        modelTestData = node.testSample(self.rawTestSamples).values
        subAucs, testY, predictList = multilModelPredictAUC_(node.models, modelTestData)
        baseAucs, _, basePredictList = multilModelPredictAUC_(self.rootNodeModels, modelTestData)
        subAuc, baseAuc = round(np.mean(subAucs), 4), round(np.mean(baseAucs), 4)
        dValue, pValue = round(subAuc - baseAuc, 4), round(tTestPValue(subAucs, baseAucs), 4)
        predict, basePredict = np.mean(predictList, axis=0), np.mean(basePredictList, axis=0)
        subF1score, baseF1score, f1DValue, subRecall, baseRecall, recallDValue, subPrecision, basePrecision, precDValue \
            = self.__showMetric(testY, predict, basePredict) 

        subNodeIndex = ""
        type = "leaf" if node.isLeaf else "inner"
        if not node.isLeaf:
            for subNode in node.subNodes:
                subNodeIndex = subNodeIndex + "-" + str(subNode.index)
        else:
            self.Y = np.hstack((self.Y, testY))
            self.predict = np.hstack((self.predict, predict))
            self.basePredict = np.hstack((self.basePredict, basePredict))

        # cmpute then save feature rank and p-value
        trainSample = node.trainSample(self.rawSamples)
        kBestFeature, pValues = getKBestFeatureSample(trainSample, len(trainSample.columns) - 1)
        aCol, rate = [node.index, node.isLeaf], []
        for size in self.savedFeatureSizeTable:
            rate.append(round(len(set(kBestFeature[:size]) & set(self.kBestFeature[:size])) / size, 4))
        aCol.extend(rate)
        aCol.extend(kBestFeature)
        self.importFeatureTable = pd.concat((self.importFeatureTable, pd.DataFrame(aCol)), axis=1)
        aCol = [node.index, node.isLeaf]
        aCol.extend(pValues)
        self.featurePValueTable = pd.concat((self.featurePValueTable, pd.DataFrame(aCol)), axis=1)

        # print subgroup info
        info = ("{} node {}, back to {}, {}, size({}, {}), {}, auc({}, {}, {}, {}), f1({}, {}, {}), recall({}, {}, {}), prec({}, {}, {}), rate {}")\
            .format(type, node.index, node.backNodeIndex, subNodeIndex, 
                    len(node.trainSampleIndex), len(node.testSampleIndex), 
                    node.splitFeature, 
                    subAuc, baseAuc, dValue, pValue,  
                    subF1score, baseF1score, f1DValue, 
                    subRecall, baseRecall, recallDValue,
                    subPrecision, basePrecision, precDValue,
                    rate)
        if isLeaf: self.leafNodeInfoList.append(info)
        else: self.innerNodeInfoList.append(info)
        self.logger.info(info)

    def __showMetric(self, testY, predict, basePredict):
        predict, basePredict = np.round(predict), np.round(basePredict)
        subF1Score, baseF1Score = f1_score(testY, predict), f1_score(testY, basePredict)
        subRecall, baseRecall = recall_score(testY, predict), recall_score(testY, basePredict)
        subPrecision, basePrecision = precision_score(testY, predict), precision_score(testY, basePredict)
        f1DValue = round(subF1Score - baseF1Score, 4)
        recallDValue = round(subRecall - baseRecall, 4) 
        precDValue = round(subPrecision - basePrecision, 4)
        return round(subF1Score, 4), round(baseF1Score, 4), f1DValue, \
               round(subRecall, 4), round(baseRecall, 4), recallDValue, \
               round(subPrecision, 4), round(basePrecision, 4), precDValue
