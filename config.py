import importlib
if importlib.util.find_spec('comet_ml'):
    from comet_ml import Experiment
from utils import get_host, get_user, C
import argparse
import torch

parser = argparse.ArgumentParser()
COMET_ML_APP_KEY = 'XR1bHdikUQC1zRRwtNaQP2huV'

"""
Most Relevant
"""

debug = False
gpu = -1 if "ken" not in get_user() else -1
use_comet_ml = False if importlib.util.find_spec('comet_ml') and not debug else False
parser.add_argument('--use_comet_ml', default=use_comet_ml)

if use_comet_ml:
    parser.add_argument('--comet_api_key', default=COMET_ML_APP_KEY)

"""
Data.
"""

""" 
dataset:
    twitter_asian_prejudice
"""
# dataset = 'twitter_asian_prejudice'
# dataset = 'r8_presplit'
dataset = 'ag_presplit'
# dataset = 'twitter_asian_prejudice_small'
if 'twitter_asian_prejudice' in dataset or 'ag' in dataset:
    num_labels = 4
elif 'r8' in dataset:
    num_labels = 8

parser.add_argument('--dataset', default=dataset)
parser.add_argument('--random_seed', default=3)


"""
Model. Pt1
"""

model = "text_gcn"

model_params = {}
parser.add_argument('--init_type', default='one_hot_init')
if model == 'text_gcn':
    n = '--model'
    pred_type = 'softmax'
    node_embd_type = 'gcn'
    layer_dim_list = [200, num_labels]
    num_layers = len(layer_dim_list)
    class_weights = False
    dropout = True
    s = 'TextGNN:pred_type={},node_embd_type={},num_layers={},layer_dim_list={},act={},' \
        'dropout={},class_weights={}'.format(
        pred_type, node_embd_type, num_layers, "_".join([str(i) for i in layer_dim_list]), 'relu', dropout, class_weights
    )
    model_params = {
        'pred_type': pred_type,
        'node_embd':  node_embd_type,
        'layer_dims': layer_dim_list,
        'class_weights': class_weights,
        'dropout': dropout
    }
    parser.add_argument(n, default=s)
else:
    raise NotImplementedError

print("{}: {}\n".format(model, model_params))
"""
Sampling
"""
word_window_size = 10
parser.add_argument('--word_window_size', default=word_window_size)
validation_window_size = 20

"""
Validation
"""
parser.add_argument("--validation_window_size", default=validation_window_size)
parser.add_argument("--validation_metric", default="loss",
                    choices=["f1_weighted", "accuracy", "loss"])
# iters_per_validation = -1
iters_per_validation = 100 if not debug else 5
parser.add_argument("--iters_per_validation", default=iters_per_validation) # if -1 then based on epochs

use_best_val_model_for_inference = True
parser.add_argument('--use_best_val_model_for_inference', default=use_best_val_model_for_inference)

"""
Evaluation.
"""
tvt_ratio = [0.8, 0.1, 0.1]
parser.add_argument('--tvt_ratio', default=tvt_ratio)
parser.add_argument('--tvt_list', default=["train", "test", "val"])
# max_eval_pairs = None


"""
Optimization.
"""

lr = 2e-2
parser.add_argument('--lr', type=float, default=lr)


device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)

# num_epochs = 100
# parser.add_argument('--num_epochs', type=int, default=num_epochs)


num_epochs = 200
num_epochs = 2 if debug else num_epochs
parser.add_argument('--num_epochs', type=int, default=num_epochs)

print_every_epochs = 5
parser.add_argument('--print_every_epochs', type=int, default=print_every_epochs)

save_model = False
parser.add_argument('--save_model', type=bool, default=save_model)
load_model = None
parser.add_argument('--load_model', default=load_model)


"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()

COMET_EXPERIMENT = None
if FLAGS.use_comet_ml:
    hyper_params = vars(FLAGS)
    COMET_EXPERIMENT = Experiment(api_key=COMET_ML_APP_KEY, project_name="textgcn")
    COMET_EXPERIMENT.log_parameters(hyper_params)
    COMET_EXPERIMENT.log_parameters(model_params)
    print("Experiment url, ", COMET_EXPERIMENT.url)
    COMET_EXPERIMENT.add_tag(FLAGS.dataset)



