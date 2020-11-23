import os
import sys
import model
import pickle
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb

from scipy.stats import levene
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold

class ClusterMethod:
    agglomerate = "Agglomerate"
    dbscan = "DBSCAN"
    kmeans = "KMeans"
class Constant:
    cluster = ClusterMethod()
    home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou/predictAKIBySubgroup"
    loggerDir = os.path.join(home, "resources", "log", "experiment-v0.6.0-log")
    trainDataDir = os.path.join(home, "resources", "data", "trainData")
    subgroupInfoSize = 4
    modelSite = 0
    dataSite = 1
    aucsSite = 2
    cvSite = 3
CONSTANT = Constant()

def getLogger(logFilePath=None, allLogPath=None):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    ALL_LOG_FORMAT = "%(asctime)s - %(thread)d - %(funcName)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)
    allLogFormatter = logging.Formatter(ALL_LOG_FORMAT)
    logger = logging.getLogger(__name__)#.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    if logFilePath is not None:
        fileHandler = logging.FileHandler(logFilePath, "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
    if allLogPath is not None:
        fileHandler2 = logging.FileHandler(allLogPath, "a")
        fileHandler2.setFormatter(allLogFormatter)
        fileHandler2.setLevel(logging.INFO)
        logger.addHandler(fileHandler2)
    return logger
# ==================================================================================================
# ============================================== 获取特征 ===========================================
def getSelectedSplitFeature(features): 
    """获取所选的暴露易感特征（80多个）"""
    splitFeature = []
    demoSplitFeature = [
        'DEMO_Age', 'DEMO_Hispanic_Y', 'DEMO_Hispanic_N', 'DEMO_Hispanic_NI',
        'DEMO_Race_01', 'DEMO_Race_02', 'DEMO_Race_03', 'DEMO_Race_04',
        'DEMO_Race_05', 'DEMO_Race_06', 'DEMO_Race_07', 'DEMO_Race_NI',
        'DEMO_Race_OT', 'DEMO_Sex_F'
    ]
    ccsSplitFeature = []
    ccsSplitFeature_num = [1, 6, 49, 50 ,59, 151, 156, 157, 158, 186, 223, 2604]
    ccsSplitFeature_num.extend([i for i in range(96, 109)])
    ccsSplitFeature_num.extend([i for i in range(122, 133)])
    ccsSplitFeature_num.extend([i for i in range(11, 48)])
    for number in ccsSplitFeature_num:
        ccsSplitFeature.append("CCS_" + str(number))
    splitFeature.extend(demoSplitFeature)
    splitFeature.extend(ccsSplitFeature)
    splitFeature = list(set(splitFeature) & set(features))
    return splitFeature

def getSelectedSplitFeature(features): 
    """获取所选的暴露易感特征（70多个，排除一部分）"""
    splitFeature = []
    demoSplitFeature = [
        'DEMO_Age', 'DEMO_Hispanic_Y', 'DEMO_Hispanic_N', 'DEMO_Hispanic_NI',
        'DEMO_Race_01', 'DEMO_Race_02', 'DEMO_Race_03', 'DEMO_Race_04',
        'DEMO_Race_05', 'DEMO_Race_06', 'DEMO_Race_07', 'DEMO_Race_NI',
        'DEMO_Race_OT', 'DEMO_Sex_F'
    ]
    ccsSplitFeature = []
    excuteFeature_num = [157, 100, 102, 104, 105, 106, 107, 1, 123, 125, 126, 129, 130, 131, 132, 151, 46, 47]
    ccsSplitFeature_num = [1, 6, 49, 50 ,59, 151, 156, 157, 158, 186, 223, 2604]
    ccsSplitFeature_num.extend([i for i in range(96, 109)])
    ccsSplitFeature_num.extend([i for i in range(122, 133)])
    ccsSplitFeature_num.extend([i for i in range(11, 48)])
    ccsSplitFeature_num = list(set(ccsSplitFeature_num) - set(excuteFeature_num))
    for number in ccsSplitFeature_num:
        ccsSplitFeature.append("CCS_" + str(number))
    splitFeature.extend(demoSplitFeature)
    splitFeature.extend(ccsSplitFeature)
    splitFeature = list(set(splitFeature) & set(features))
    return splitFeature

def getSelectedSplitFeature_v3(features): 
    """获取所选的暴露易感特征（全部不可变特征，排除一部分）"""
    results = []
    excuteFeature_num = [157, 100, 102, 104, 105, 106, 107, 1, 123, 125, 126, 129, 130, 131, 132, 151, 46, 47, 259]
    excuteFeature = []
    for number in excuteFeature_num:
        excuteFeature.append("CCS_" + str(number))
    excuteFeature.extend(['DEMO_Sex_F'])
    for feature in features:
        prefix = feature[: 2]
        if prefix == "DE" or prefix == "CC":
            results.append(feature)
    results_set = set(results)
    excuteFeature_set = set(excuteFeature)
    results = list(results_set - excuteFeature_set)
    return results

def getAllSplitFeature(features):
    results = []
    for feature in features:
        prefix = feature[: 2]
        if prefix == "DE" or prefix == "CC":
            results.append(feature)
    return results

def getKBestFeatureSample(samples, k=800):
    """get best k feature by chi2"""
    X, Y = samples.values[:, :-1], samples.values[:, -1]
    model = SelectKBest(chi2, k=k)
    X = model.fit_transform(X, Y)
    scores, pValues = model.scores_, model.pvalues_
 
    indices = np.argsort(scores)[::-1] # 将scores按照升序排列后逆转
    k_best_features = list(samples.columns.values[indices[0:k]])
    k_best_features.append("Label")
    
    k_best_features_pValue = list(pValues[indices[0:k]])
    k_best_features_pValue.append(-1)
    return k_best_features, k_best_features_pValue

def getKFixedFeature(samples, k):
    allFixedFeatures = getAllSplitFeature(samples.columns)
    return getKBestFeature(samples[allFixedFeatures], k)

def getKSelectedFeature(samples, k):
    selectedFeatures = getSelectedSplitFeature(samples.columns)
    if k < 0:
        return selectedFeatures
    return getKBestFeature(samples[selectedFeatures], k) 

def getSelectedTrainFeature(features):
    # lab, vital, and med
    results = []
    for feature in features:
        prefix = feature[:3]
        if prefix == "LAB" or prefix == "VIT" or prefix == "MED":
            results.append(feature)
    return results

# ==================================================================================================
# =============================================== 数据处理 ==========================================
def trainTestSamplesSplit(samples, testSize=0.3, isStratified=True):
    trainSamples, testSamples = [], []
    if isStratified:
        posSamples = samples[samples["Label"]==1.0]
        negSamples = samples[samples["Label"]==0.0]
        testPosSamples = posSamples.sample(frac=testSize, replace=False, axis=0)
        testNegSamples = negSamples.sample(frac=testSize, replace=False, axis=0)
        testSamples = pd.concat([testPosSamples, testNegSamples], axis=0)
        trainSampleIndex = set(samples.index) - set(testSamples.index) 
        trainSamples = samples.loc[trainSampleIndex]
    else:
        testSamples = samples.sample(frac=testSize, replace=False, axis=0)
        trainSamples = samples.drop(testSamples.index)
    return trainSamples.sample(frac=1), testSamples.sample(frac=1)

# ==================================================================================================
# =============================================== 建模方法 ==========================================
def trainModelTransferModel(trainData, initModels):
    modelList = []
    params = {
        'task': 'train',
        'application': 'binary',  # 目标函数
        'boosting_type': 'gbdt',  # 设置提升类型
        'learning_rate': 0.01,  # 学习速率
        'min_data_in_leaf': 100,
        'metric': ['auc'],
        'num_trees': 100,
        'verbose': -1
    }
    X, Y = trainData[:, :-1], trainData[:, -1]
    for initModel in initModels:
        lgb_train = lgb.Dataset(X, Y)
        clf = lgb.train(params, train_set=lgb_train, num_boost_round=100,
                        init_model=initModel, keep_training_booster=True)
        modelList.append(clf)
    return modelList

def sampleThenTrainTransferModel(samples, initModels, size):
    modelList = []
    params = {
        'task': 'train',
        'application': 'binary',  # 目标函数
        'boosting_type': 'gbdt',  # 设置提升类型
        'learning_rate': 0.01,  # 学习速率
        'min_data_in_leaf': 100,
        'metric': ['auc'],
        'num_trees': 100,
        'verbose': -1
    }
    for initModel in initModels:
        trainData = samples.sample(n = size).values
        X, Y = trainData[:, :-1], trainData[:, -1]
        lgb_train = lgb.Dataset(X, Y)
        clf = lgb.train(params, train_set=lgb_train, num_boost_round=100,
                        init_model=initModel, keep_training_booster=True)
        modelList.append(clf)
    return modelList

def trainModelThenPredict(trainData, testData, returnModel = False):
    testX, testY = testData[:, :-1], testData[:, -1]
    trainX, trainY = trainData[:, :-1], trainData[:, -1]
    params = {
        'task': 'train',
        'application': 'binary',  # 目标函数
        'boosting_type': 'gbdt',  # 设置提升类型
        'learning_rate': 0.01,  # 学习速率
        'min_data_in_leaf': 100,
        'metric': {'auc'},  
        'num_trees': 100,
        'verbose': -1
    }
    lgb_train = lgb.Dataset(trainX, trainY)
    clf = lgb.train(params, train_set=lgb_train, num_boost_round=100)
    predicts = clf.predict(testX)
    if np.sum(testY) == 0 or np.sum(testY) == len(testY) or \
        np.sum(predicts) == 0 or np.sum(predicts) == len(predicts):
        return 0
    auc = round(roc_auc_score(testY, predicts), 4)
    if returnModel: return auc, clf
    else: return auc

def sampleThenPredict(trainSample, testSample, targetSize, time=10):
    aucList = []
    for _ in range(time):
        noAkiSize = 0
        while noAkiSize == 0 or noAkiSize == targetSize:
            sample = trainSample.sample(n=targetSize)
            noAkiSize = np.sum(sample.values[:, -1] == 0)
        aucList.append(trainModelThenPredict(sample.values, testSample.values))
    return round(np.mean(aucList), 4)

def trainLightgbmCV(trainData, cv=10):
    params = {
        'task': 'train',
        'application': 'binary',  # 目标函数
        'boosting_type': 'gbdt',  # 设置提升类型
        'learning_rate': 0.01,  # 学习速率
        'min_data_in_leaf': 100,
        'metric': {'auc'},  
        'num_trees': 100,
        'verbose': -1
    }
    modelList = []
    X, Y = trainData[:, :-1], trainData[:, -1]
    kflod = StratifiedKFold(n_splits=cv, shuffle=True)
    for trainIndex, _ in kflod.split(X, Y):
        trainX, trainY = X[trainIndex], Y[trainIndex]
        lgb_train = lgb.Dataset(trainX, trainY)
        clf = lgb.train(params, train_set=lgb_train, num_boost_round=100)
        modelList.append(clf)
    return modelList

def multilModelPredictAUC(subModels, testData):
    aucList = []
    testX, testY = testData[:, :-1], testData[:, -1]
    for aModel in subModels:
        predicts = aModel.predict(testX)
        aucList.append(roc_auc_score(testY, predicts))
    return aucList

def multilModelPredictAUC_(subModels, testData):
    aucList, predictList = [], []
    testX, testY = testData[:, :-1], testData[:, -1]
    for aModel in subModels:
        predicts = aModel.predict(testX)
        aucList.append(roc_auc_score(testY, predicts))
        predictList.append(predicts)
    return aucList, testY, predictList
"""
def trainModelThenPredict(trainData, testData):
    testX, testY = testData[:, :-1], testData[:, -1]
    trainX, trainY = trainData[:, :-1], trainData[:, -1]
    clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=100, boosting_type="gbdt",
        objective="binary").fit(trainX, trainY, eval_metric='binary_logloss')
    predicts = clf.predict_proba(testX)
    if np.sum(testY) == 0 or np.sum(testY) == len(testY) or \
        np.sum(predicts[:, 1]) == 0 or np.sum(predicts[:, 1]) == len(predicts[:, 1]):
        return 0
    return roc_auc_score(testY, predicts[:, 1])

def sampleThenPredict(trainSample, testSample, targetSize, time=10):
    aucList = []
    trainData, testData, cv = trainSample.values, testSample.values, time
    testX, testY = testData[:, :-1], testData[:, -1]
    for _ in range(cv):
        noAkiSize = 0
        while noAkiSize == 0 or noAkiSize == targetSize:
            data = pd.DataFrame(trainData).sample(n=targetSize).values
            noAkiSize = np.sum(data[:, -1] == 0)
        aucList.append(trainModelThenPredict(data, testData))
    return round(np.sum(aucList) / cv, 4)

def trainLightgbmCV(trainData, cv=10):
    param = { 
        "learning_rate": 0.1, "n_estimators": 100, "boosting_type": "gbdt", 
        "objective": "binary", "random_state": 1 
    }
    modelList = []
    X, Y = trainData[:, :-1], trainData[:, -1]
    kflod = StratifiedKFold(n_splits=cv, random_state=123, shuffle=True)
    for trainIndex, _ in kflod.split(X, Y):
        trainX, trainY = X[trainIndex], Y[trainIndex]
        clf = lgb.LGBMClassifier(**param).fit(trainX, trainY, eval_metric='binary_logloss')
        modelList.append(clf)
    return modelList

def multilModelPredictAUC(subModels, testData):
    aucList = []
    testX, testY = testData[:, :-1], testData[:, -1]
    for aModel in subModels:
        predicts = aModel.predict_proba(testX)
        aucList.append(roc_auc_score(testY, predicts[:, 1]))
    return aucList
"""
# ========================================== 结果验证 ===============================================
def tTestPValue(a, b):
    equalVal = (levene(a, b).pvalue < 0.05) # 如果p值大�?.05，说明具有方差齐性，需要将参数置为False
    return ttest_ind(a, b, equal_var=equalVal).pvalue