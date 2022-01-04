import pickle

from pandas.core.frame import DataFrame

from pub_class import *
from pub_function import *
from metric import metric_auc
from metric import metric_cart
from metric import metric_ghhi


class SubgroupIdentifyTree:
    def __init__(self, tree_param, candidates, metric='ghhi', root_model_param=None,
                 other_param=None, train_transfer_model=False, metric_func=None, debug_log=True):
        self.__init_params()
        self.__init_tree_params(tree_param)
        self.__init_metric_func(metric, metric_func)
        self.candidates = candidates
        self.train_transfer_model = train_transfer_model
        self.root_model_param = root_model_param
        self.other_param = other_param
        self.__debug_log = debug_log

    def __init_params(self):
        pass

    def __init_tree_params(self, tree_param):
        pass

    def __init_metric_func(self, metric, metric_func):
        pass

    def fit(self, data):
        pass

        return self

    def __createTree(self, node):
        pass

    def __get_qualified_split(self, p_node):
        pass

    def __next_index(self):
        pass

    def __generate_subgroup(self, data, split_point, conditions):
        pass

    def __show_node_info(self, node):
        pass

    def identify(self, data:DataFrame) -> dict:
        pass

    def predict_subgroup(self, node_index, data):
        pass

    def predict_subgroups(self, subgroups):
        pass

    def predict(self, data:DataFrame):
        pass

    def subgroup_info(self, index):
        pass

    def __split(self, data, definitions):
        pass

    def train_model_for_subgroup(self):
        pass

    def save_subgroup_shap(self):
        pass

    def save_feature_importance(self):
        pass