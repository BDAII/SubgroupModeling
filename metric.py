import numpy as np

from pub_function import *
from sklearn.metrics import roc_auc_score

def metric_auc(sub_datas, root_model, p_node, other_params={}):
    metric = 0.0

    all_size = np.sum([len(data) for data in sub_datas])
    for data in sub_datas:
        train, test = stratifiedSplit(data, test_rate=0.3)
        model = simple_search_and_train_gbm(train.values)
        x_test, y_test = test.values[:, :-1], test['Label']
        auc = roc_auc_score(y_test, model.predict(x_test))
        weight = len(data) / all_size
        metric += (auc * weight)
        # debug('split node {}, auc {}, weight {}'.format(p_node.index, round(auc, 4), round(weight, 4)))
    
    return metric


def metric_ghhi(sub_datas, root_model, p_node, other_params={}):
    metric = 0.0

    all_size = np.sum([len(data) for data in sub_datas])
    for data in sub_datas:
        train, test = stratifiedSplit(data, test_rate=0.3)
        model = simple_search_and_train_gbm(train.values)
        ghhi = compute_ghhi_with_gbm(model, test, other_params['ghhi_feature'])
        weight = len(data) / all_size
        metric += (ghhi * weight)

    return metric


def metric_cart(sub_datas, root_model, p_node, other_params={}):
    pass

def metric_ghhi_with_transfer_model(sub_datas, root_model, node, other_params={}):
    pass

def metric_auc_with_transfer_model(sub_datas, root_model, node, other_params={}):
    pass