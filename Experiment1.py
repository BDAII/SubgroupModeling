# subgroup identify
import os
import sys 

from pub_function import *
from subgroup_identify import SubgroupIdentifyTree

# 加载数据，并预处理（特征筛选）
data_file = sys.argv[1]
data_path = os.path.join(DATA_DIR, data_file) 
data = pd.read_pickle(data_path).sample(n=3000)                                         ; show_log("data: {}".format(data.shape))

tree_param = {
    'depth': 3,
    'min_split_size': 1000,
    'min_leaf_size': 300
}
root_model_param = {
    'max_depth': 15,
    'learning_rate': 0.05,
    'feature_fraction': 1
}
candidates = get_top_k_feature(data, 20)
debug("get candidates {}...".format(candidates[:10]))
other_param = {
    'ghhi_feature': get_top_k_feature(data, 300)
}

tree = SubgroupIdentifyTree(tree_param=tree_param, candidates=candidates, metric='ghhi', 
                            other_param=other_param, root_model_param=root_model_param)
tree.fit(data)

save_path = os.path.join(MODEL_DIR, "tree_{}.pkl".format(cur_time_stamp()))
save_to_local(tree, save_path)
debug('model saved to local: {}'.format(save_path))

