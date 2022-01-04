# build model for subgroup
import os
import sys

import pandas as pd

from pub_funtion import * 

model_path = os.path.join(MODEL_DIR, sys.argv[1])
test_path = os.path.join(DATA_DIR, sys.argv[2])
tree = load_from_local(model_path)
test = pd.read_pickle(test_path)

cv = 10
result_map = {}
for i in range(cv):
    test_cv = test.sample(frac=1, replace=True)
    cur_node_map = tree.identify(test_cv)

    for node_index, samples in cur_node_map.items():
        data = test.loc[samples]
        y_true, y_pred = tree.predict_subgroup(node_index, data)

        not_init = (result_map[node_index] == None)
        if not_init:
            result_map[node_index] = []
        result_map[node_index].append({'y_true': y_true, 'y_pred': y_pred, 'test_data': data})

show_metric(result_map, tree.root_model, [show_auc, show_recall])
tree.save_subgroup_shap()
tree.save_feature_importance()

