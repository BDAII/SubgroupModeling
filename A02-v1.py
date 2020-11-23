import os
import sys
import model
import pickle
import traceback
import numpy as np
import pandas as pd

from util import *
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import pearsonr

def loadAndSplitSamples(trainFile, testFile):
    trainSample = pd.read_pickle(os.path.join(Constant.trainDataDir, trainFile))
    testSample = pd.read_pickle(os.path.join(Constant.trainDataDir, testFile))
    return trainSample, testSample

logger = getLogger()
trainSample, testSample = loadAndSplitSamples(sys.argv[1], sys.argv[2])
bestFixedFeatures = getKSelectedFeature(trainSample, -1)
logger.info ("selected fixed feature {}".format(len(bestFixedFeatures)).center(80, "-"))
logger.info(bestFixedFeatures)
logger.info("train sample {}, test sample {}".format(trainSample.shape, testSample.shape).center(80, "-"))

"""feature select by pearson"""
features = bestFixedFeatures
targetFeatures = []
# get each feature's pearson correlation coefficient with label
labels = trainSample["Label"]
pTable = {}
for feature in features:
    pTable[feature] = pearsonr(trainSample[feature].values, labels)[0]
# iterat each feature, delete the feature which has hige pearson correlation coefficient
for feature in features:
    maxPearson, maxPearsonFeature = -1, None
    for otherFeature in features:
        pearson = pearsonr(trainSample[feature].values, trainSample[otherFeature])[0]
        if pearson > maxPearson:
            maxPearson, maxPearsonFeature = pearson, otherFeature
    if maxPearson <= 0.9: targetFeatures.append(feature)
    else:
        savedFeature, deletedFeature = feature, otherFeature
        featurePearson, otherFeaturePearson = pTable[feature], pTable[otherFeature]
        if featurePearson < otherFeaturePearson: 
            savedFeature, deletedFeature = otherFeature, feature
        while deletedFeature in targetFeatures: targetFeatures.remove(deletedFeature)
        targetFeature.append(savedFeature) 
logger.info ("after delete high correlate feature {}".format(len(targetFeatures)).center(80, "-"))
logger.info(targetFeatures)
minSamplesLeaf, minSampleSplit = 500, 1500
param = {
    "minSamplesLeaf": minSamplesLeaf, 
    "minSampleSplit": minSampleSplit, 
    "maxDepth": 50, 
    "fixedFeature": targetFeatures,
    "logger": logger
}
logger.info("{}-{}".format(minSamplesLeaf, minSampleSplit))
tree = model.DecisionTree(**param).fitThenPredict(trainSample, testSample)


