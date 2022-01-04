import os
import pickle
import shap
import logging
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb

# from pub_class import *

from scipy import stats
from scipy.stats import levene
from scipy.stats import ttest_ind
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from pandas.core.frame import DataFrame


def cur_time_stamp():
    today = datetime.today()
    return "{}{:02d}{:02d}-{:02d}{:02d}".format(
        str(today.year)[-2:],
        today.month,
        today.day,
        today.hour,
        today.second)


def getLogger():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)

    logger = logging.getLogger(__name__)  # .setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    file_handler = logging.FileHandler('log-{}.log'.format(cur_time_stamp()))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def show_log(msg):
    LOGGER.info(msg)


def debug(msg, show=True):
    if show:
        LOGGER.debug(msg)


LOG_SEARCH_GBM = True
LOG_SPLIT = True
LOG_COMPUTE_GHHI = True
LOGGER = getLogger()
KU_HOME = "***"
PROJECT_DIR = os.path.join(KU_HOME, "GHHI", "SubgroupIdentify")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
SH_DIR = os.path.join(PROJECT_DIR, "sh")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
# =====================================================================================
# ================================== function =========================================
# =====================================================================================


def submit_new_task(task_path):
    command = "sbatch {}".format(task_path)
    os.system(command)


def entropy(Y):
    size = len(Y)
    pos_size = np.sum(Y == 1)
    neg_size = size - pos_size
    pos_rate, neg_rate = float(pos_size) / float(size), float(neg_size) / float(size)
    return -(pos_rate * np.log2(pos_rate) + neg_rate * np.log2(neg_rate))


def get_best_gini(data, feature):
    Y, T = np.array(data.iloc[:]["Label"]), []
    feature_values = data[feature]
    unique_values = np.sort(list(set(feature_values)))
    D, ent_Y, max_gain, best_gini = len(Y), entropy(Y), -999999, 0

    for i in range(1, len(unique_values)):
        t = float(unique_values[i - 1] + unique_values[i]) / 2.0
        Dtp, Dtn = Y[feature_values >= t], Y[feature_values < t]
        if len(Dtp) == 0 or len(Dtn) == 0:
            continue
        gain = ent_Y - (len(Dtp) / D * entropy(Dtp) + len(Dtn) / D * entropy(Dtn))
        if gain > max_gain:
            max_gain, best_gini = gain, t
    return best_gini


def get_top_k_feature(data, k, return_pvalue=False):
    """get best k feature by chi2"""
    X, Y = data.values[:, :-1], data.values[:, -1]
    model = SelectKBest(chi2, k=k)
    X = model.fit_transform(X, Y)
    scores, pValues = model.scores_, model.pvalues_

    indices = np.argsort(scores)[::-1]  # 将scores按照升序排列后逆转
    k_best_features = list(data.columns.values[indices[0: k]])
    k_best_features_pValue = list(pValues[indices[0: k]])

    if return_pvalue:
        return k_best_features, k_best_features_pValue
    else:
        return k_best_features


def train_test_split(data, cv):
    train, test = [], []

    k_fold = StratifiedKFold(n_splits=cv, random_state=0, shuffle=True)
    X, Y = data.index, data.values[:, -1]
    for train_index, test_index in k_fold.split(X, Y):
        train.append(X[train_index])
        test.append(X[test_index])
    return zip(train, test)


def stratifiedSample(data, pos, neg):
    pos_index = (data['Label'] == 1)
    neg_index = (data['Label'] == 0)
    pos_sample = data.loc[pos_index].sample(n=pos)
    neg_sample = data.loc[neg_index].sample(n=neg)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1)


def stratifiedSplit(data, test_rate=0.3): 
    pos_sample = data.loc[data['Label'] == 1]
    neg_sample = data.loc[data['Label'] == 0]

    test_pos = pos_sample.sample(frac=test_rate)
    test_neg = neg_sample.sample(frac=test_rate)
    test = pd.concat([test_pos, test_neg]).sample(frac=1)

    train_pos = pos_sample.loc[list(set(pos_sample.index) - set(test_pos.index))]
    train_neg = neg_sample.loc[list(set(neg_sample.index) - set(test_neg.index))]
    train = pd.concat([train_pos, train_neg]).sample(frac=1)
    return train, test


def is_discrete_feature(data, feature):
    values = list(set(data[feature]))

    is_binary_feature = (len(values) <= 2)
    if is_binary_feature:
        return True
    return False


def build_gbm_model(data, p=None):
    n_tree = 100 if p is None or p['n_tree'] is None else p['n_tree']
    params = {
        'task'             : 'train',
        'application'      : 'binary',  # 目标函数
        'boosting_type'    : 'gbdt',    # 设置提升类型
        'max_depth'        : 4 if p is None or p['depth'] is None else p['depth'],
        'num_leaves'       : 31 if p is None or p['leaves'] is None else p['leaves'],
        'learning_rate'    : 0.01,      # 学习速率
        'min_data_in_leaf' : 3 if p is None or p['leaf'] is None else p['leaf'],
        'metric'           : {'auc'},
        'verbose'          : -1
    }

    X, Y = data[:, :-1], data[:, -1]
    lgb_train_data = lgb.Dataset(X, Y)
    return lgb.train(params, train_set=lgb_train_data, num_boost_round=n_tree)


def compute_auc(models, test_datas):
    if len(models) != len(test_datas):
        raise Exception("model num don't match test data num!")

    Y_true, Y_pred = [], []
    for model, test_data in zip(models, test_datas):
        Y_true.extend(test_data[:, -1])
        Y_pred.extend(model.predict(test_data))
    return roc_auc_score(Y_true, Y_pred)


def compute_feature_importance(shap_values, data, features):
    result = pd.DataFrame()

    pos_sample_index = data.index[data['Label'] == 1]
    neg_sample_index = data.index[data['Label'] == 0]
    for feature in features:
        result.loc[feature, 'value'] = shap_values.loc[pos_sample_index, feature].mean(axis=0) - \
                                       shap_values.loc[neg_sample_index, feature].mean(axis=0)
    return result


def compute_ghhi_with_gbm(model, test_data:DataFrame, features):
    debug('[compute_ghhi_with_gbm] compute ghhi...', LOG_COMPUTE_GHHI)
    shap_values = shap.TreeExplainer(model).shap_values(test_data.values[:, :-1])[1]
    debug('[compute_ghhi_with_gbm] get shap value...', LOG_COMPUTE_GHHI)
    shap_values = pd.DataFrame(shap_values)
    shap_values.columns = test_data.columns[:-1]
    shap_values.index = test_data.index

    debug('[compute_ghhi_with_gbm] get feature importance...', LOG_COMPUTE_GHHI)
    feature_importance = compute_feature_importance(shap_values, test_data, features)
    debug('[compute_ghhi_with_gbm] get corr data...', LOG_COMPUTE_GHHI)
    corr_data = test_data[features].corr().abs()
    return compute_ghhi(feature_importance, corr_data)


def compute_ghhi(feature_importance, corr_data):
    debug('[compute_ghhi_with_gbm] compute ghhi(feature {})...'.format(feature_importance.shape), LOG_COMPUTE_GHHI)
    a = np.sum(feature_importance.values ** 2)
    b = 0
    for f_i in feature_importance.index:
        for f_j in feature_importance.index:
            if f_i == f_j:
                continue
            b += (np.abs(corr_data.loc[f_i, f_j]) *
                  feature_importance.loc[f_i, 'value'] *
                  feature_importance.loc[f_j, 'value'])
    c = np.sum(feature_importance.values) ** 2
    return (a + b) / c


def save_to_local(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_from_local(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def search_and_train_gbm(data, init_model=None):
    search_params = {
        'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
        'depth': [3, 5, 6, 7, 9, 12, 15],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1]
    }
    params = search_gbm(data, search_params, cv=5, init_model=init_model)
    return train_gbm(data, params)


def simple_search_and_train_gbm(data, init_model=None):
    debug('[simple search] search for data {}'.format(data.shape), LOG_SEARCH_GBM)
    search_params = {
        'learning_rate': [0.01, 0.05],
        'depth': [3, 6],
        'feature_fraction': [0.7, 0.9]
    }
    params = search_gbm(data, search_params, cv=3, init_model=init_model)
    debug('[simple search] search over, retrain model...', LOG_SEARCH_GBM)
    return train_gbm(data, params)


def search_and_train_transfer_gbm(data, init_model):
    return search_and_train_gbm(data, init_model)


def simple_search_and_train_transfer_gbm(data, init_model):
    return simple_search_and_train_gbm(data, init_model)


def search_gbm(data, search_params, cv=5, init_model=None):
    X, Y = data[:, :-1], data[:, -1]
    kfold = StratifiedKFold(n_splits=cv)

    max_auc = 0
    best_depth, best_learning_rate , best_feature_fraction = 0, 0, 0
    for learning_rate in search_params['learning_rate']:
        for depth in search_params['depth']:
            for feature_fraction in search_params['feature_fraction']:
                p = {
                    'task': 'train',
                    'objective': 'binary',
                    'boosting': 'gbdt',  # 设置提升类型
                    'metric': {'auc'},
                    'verbose': -1,
                    'max_depth': depth,
                    'learning_rate': learning_rate,
                    'feature_fraction': feature_fraction
                }

                y_true, y_score = [], []

                for train_index, test_index in kfold.split(X, Y):
                    x_train, y_train = X[train_index], Y[train_index]
                    x_test, y_test = X[test_index], Y[test_index]

                    train_set = lgb.Dataset(x_train, y_train)
                    model = lgb.train(params=p, train_set=train_set, init_model=init_model, num_boost_round=150)

                    y_true.extend(y_test)
                    y_score.extend(model.predict(x_test))

                auc = roc_auc_score(y_true, y_score)
                if auc > max_auc:
                    max_auc = auc
                    best_learning_rate= learning_rate
                    best_depth = depth
                    best_feature_fraction = feature_fraction
                debug("learning rate:{}, depth:{}, feature feaction {} -> auc {}"
                         .format(learning_rate, depth, feature_fraction, auc), LOG_SEARCH_GBM and LOG_SPLIT)
    debug("[BEST] learning rate:{}, depth:{}, feature feaction {} -> auc {}"
             .format(best_learning_rate, best_depth, best_feature_fraction, max_auc), LOG_SEARCH_GBM and LOG_SPLIT)

    return {
        'learning_rate': best_learning_rate,
        'max_depth': best_depth,
        'feature_fraction': best_feature_fraction
    }


def train_gbm(data, params, init_model=None):
    p = {
        'task': 'train',
        'objective': 'binary',
        'boosting': 'gbdt',     # 设置提升类型
        'learning_rate': 0.01,  # 学习速率
        'metric': {'auc'}
    }
    for k, v in params.items():
        p[k] = v

    x_train, y_train = data[:, :-1], data[:, -1]
    train_set = lgb.Dataset(x_train, y_train)
    return lgb.train(params=p, train_set=train_set, init_model=init_model, num_boost_round=150)


def ttest_pvalue(a, b):
    equalVal = (levene(a, b).pvalue < 0.05)
    return ttest_ind(a, b, equal_var=equalVal).pvalue


def show_metric(result_map, root_model, show_func_list):
    subgroups_auc_map, base_auc_map = {}, {}
    for node_index, result_list in result_map.items():
        show_log(' node {} metric result '.format(node_index).center(80, '='))
        y_true_list = [result['y_true'] for result in result_list]
        y_sub_pred_list = [result['y_pred'] for result in result_list]
        y_base_pred_list = [root_model.predict(result['test_data']) for result in result_list]

        subgroups_auc_map[node_index] = []
        base_auc_map[node_index] = []
        for show in show_func_list:
            sub_auc, base_auc = show(y_true_list, y_sub_pred_list, y_base_pred_list)
            subgroups_auc_map[node_index].append(sub_auc)
            base_auc_map[node_index].append(base_auc)

def show_auc(y_true_list, y_sub_pred_list, y_base_pred_list):
    show_log(' auc info '.center(40, '*'))
    sub_auc_list, base_auc_list = [], []
    for i in range(0, len(y_true_list)):
        y_true, y_sub, y_base = y_true_list[i], y_sub_pred_list[i], y_base_pred_list[i]
        sub_auc_list.append(round(roc_auc_score(y_true, y_sub), 4))
        base_auc_list.append(round(roc_auc_score(y_true, y_base), 4))
    sub_auc_ci = stats.t.interval(alpha=0.95, df=len(sub_auc_list) - 1, loc=np.mean(sub_auc_list), scale=np.std(sub_auc_list))
    base_auc_ci = stats.t.interval(alpha=0.95, df=len(base_auc_list) - 1, loc=np.mean(base_auc_list), scale=np.std(base_auc_list))
    p_value = round(ttest_pvalue(sub_auc_list, base_auc_list), 4)
    sub_auc_mean = round(np.mean(sub_auc_list), 4)
    base_auc_mean = round(np.mean(base_auc_list), 4)
    show_log('{}{}{}'.format('sub auc'.center(10, ' '), 'base auc'.center(10, ' '), 'd-value'.center(10, ' ')))
    show_log('{}{}{}'.format(str(sub_auc_mean).center(10, ' '), str(base_auc_mean).center(10, ' '), str(round(sub_auc_mean - base_auc_mean, 4)).center(10, ' ')))
    show_log('sub auc ci: {}'.format(sub_auc_ci))
    show_log('base auc ci: {}'.format(base_auc_ci))
    show_log('sub auc: {}'.format(sub_auc_list))
    show_log('base auc: {}'.format(base_auc_list))
    show_log('p-value: {}'.format(p_value))
    show_log(''.center(40, '*'))


def show_recall():
    pass

