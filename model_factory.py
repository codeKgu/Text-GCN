from config import FLAGS
from collections import Counter
from model_text_gnn import TextGNN
from utils import parse_as_int_list
import torch

def create_model(dataset):
    sp = vars(FLAGS)["model"].split(':')
    name = sp[0]
    layer_info = {}
    if len(sp) > 1:
        assert (len(sp) == 2)
        for spec in sp[1].split(','):
            ssp = spec.split('=')
            layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
    if name in model_ctors:
        return model_ctors[name](layer_info, dataset)
    else:
        raise ValueError("Model not implemented {}".format(name))


def create_text_gnn(layer_info, dataset):

    lyr_dims = parse_as_int_list(layer_info["layer_dim_list"])
    lyr_dims = [dataset.node_feats.shape[1]] + lyr_dims
    weights = None
    if layer_info["class_weights"].lower() == "true":
        counts = Counter(dataset.label_inds[dataset.node_ids])
        weights = len(counts) * [0]
        min_weight = min(counts.values())
        for k, v in counts.items():
            weights[k] = min_weight / float(v)
        weights = torch.tensor(weights, device=FLAGS.device)

    return TextGNN(
        pred_type=layer_info["pred_type"],
        node_embd_type=layer_info["node_embd_type"],
        num_layers=int(layer_info["num_layers"]),
        layer_dim_list=lyr_dims,
        act=layer_info["act"],
        bn=False,
        num_labels=len(dataset.label_dict),
        class_weights=weights,
        dropout=layer_info["dropout"]
    )


model_ctors = {
    'TextGNN': create_text_gnn,
}
