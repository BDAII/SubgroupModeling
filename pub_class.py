import numpy as np

from pub_function import *
from pandas.core.frame import DataFrame



def getLogger():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)
    logger = logging.getLogger(__name__)  # .setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def show_log(msg):
    LOGGER.info(msg)


def debug(msg, show=True):
    if show:
        LOGGER.info(msg)


LOGGER = getLogger()
DEBUG_LOG = True
# =====================================================================================
# ==================================== class ==========================================
# =====================================================================================
class Subgroup:
    TYPE_BINARY = 'binary'
    TYPE_NUMERIC = 'numeric'

    SUBGROUP_GT = 'gt'
    SUBGROUP_LT = 'lt'

    def __init__(self, samples, split_feature, split_feature_value, split_feature_type, gini_value = None):
        self.samples = samples
        self.split_feature = split_feature
        self.split_feature_value = split_feature_value
        self.split_feature_type = split_feature_type
        self.gini_value = gini_value

    def get_canonical_name(self):
        base = "{}-{}".format(self.split_feature, self.split_feature_value)
        if self.split_feature_type is Subgroup.TYPE_BINARY:
            return base
        elif self.split_feature_type is Subgroup.TYPE_NUMERIC:
            return "{}-{}".format(base, self.gini_value)
        else:
            return "UN_KNOW"
# =====================================================================================


class TreeNode:
    TYPE_LEAF = 'leaf'
    TYPE_INNER = 'inner'
    TYPE_UNKNOW = 'unknow'

    def __init__(self, samples, model, depth, index, p_index, definitions):
        self.samples, self.model, self.index = samples, model, index
        self.depth, self.p_index, self.definitions = depth, p_index, definitions

        self.type = TreeNode.TYPE_UNKNOW
        self.children = []

    def mark_to_inner(self):
        self.type = TreeNode.TYPE_INNER

    def mark_to_leaf(self):
        self.type = TreeNode.TYPE_LEAF

    def is_leaf(self):
        return self.type == TreeNode.TYPE_LEAF

    def is_inner(self):
        return self.type == TreeNode.TYPE_INNER

    def get_definition(self):
        result = ""

        for feature in self.definitions.index:
            result += "[{}-{}]_".format(feature, self.definitions.loc[feature, 'value'], )

        return result[:-1] if len(result) > 0 else result
# =================================================================================================


class SubgroupTester:
    pass


class SampleSizeTester(SubgroupTester):
    def __init__(self, pos, neg, size):
        self.min_pos_size = pos
        self.min_neg_size = neg
        self.min_size = size

    def test(self, data):
        pos_size = np.sum(data['Label'] == 1)
        neg_size = np.sum(data['Label'] == 0)
        all_size = len(data)

        debug("[SampleSizeTester.test] len {}, pos {} - {}, neg {} - {}, size {} - {} | {}"
              .format(len(data), pos_size, self.min_pos_size, neg_size, self.min_neg_size, all_size, self.min_size,
                       (pos_size >= self.min_pos_size) and (neg_size >= self.min_neg_size) and (all_size >= self.min_size)
        ), DEBUG_LOG)

        return (pos_size >= self.min_pos_size) and \
               (neg_size >= self.min_neg_size) and \
               (all_size >= self.min_size)
# =================================================================================================


class TreeNodeTester:
    pass


class NodeSampleTester(TreeNodeTester):
    def __init__(self, min_sample_size):
        self.min_sample_size = min_sample_size

    def test(self, node:TreeNode, data:DataFrame) -> bool:
        sample_size = len(node.samples)
        pos_size = np.sum(data.loc[node.samples, 'Label'] == 1)
        neg_size = np.sum(data.loc[node.samples, 'Label'] == 0)

        return (sample_size >= self.min_sample_size) and (pos_size > 0) and (neg_size > 0)


class NodeDepthTester(TreeNodeTester):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def test(self, node:TreeNode, data:DataFrame) -> bool:
        return node.depth < self.max_depth
# =================================================================================================


class Tester:
    @staticmethod
    def isValidSubgroup(data, conditions) -> bool:
        debug('[isValidSubgroup] all {}, pos {}, neg {}'.format(len(data), np.sum(data['Label'] == 1), np.sum(data['Label'] == 0)), DEBUG_LOG)
        for condition in conditions:
            if not condition.test(data):
                return False
        return True

    @staticmethod
    def canSplit(node, data, conditions) -> bool:
        for condition in conditions:
            if not condition.test(node, data):
                return False
        return True
# =================================================================================================

