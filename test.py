import numpy as np
import pickle
import json


def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

with open('/root/saba/diffusion-classifier/data/cifar10/v2-0_1trials_50_100_200samples_cluster_d3_l1_100spc/hierarchical_tree.pkl', 'rb') as f:
    tree = pickle.load(f)

tree_clean = convert_numpy_to_native(tree)

with open('tree.json', 'w') as f:
    json.dump(tree_clean, f, indent=2)



