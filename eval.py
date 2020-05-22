import numpy as np
from sklearn import metrics


def eval(preds, dataset, test=False):
    y_true = dataset.label_inds[dataset.node_ids]
    y_pred_label = [np.argmax(pred) for pred in preds]
    accuracy = metrics.accuracy_score(y_true, y_pred_label)
    f1_weighted = metrics.f1_score(y_true, y_pred_label, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred_label, average='macro')
    f1_micro = metrics.f1_score(y_true, y_pred_label, average='micro')
    precision_weighted = metrics.precision_score(y_true, y_pred_label, average='weighted')
    precision_macro = metrics.precision_score(y_true, y_pred_label, average='macro')
    precision_micro = metrics.precision_score(y_true, y_pred_label, average='micro')
    recall_weighted = metrics.recall_score(y_true, y_pred_label, average='weighted')
    recall_macro = metrics.recall_score(y_true, y_pred_label, average='macro')
    recall_micro = metrics.recall_score(y_true, y_pred_label, average='micro')
    results = {"accuracy": accuracy,
               "f1_weighted": f1_weighted,
               "f1_macro": f1_macro,
               "f1_micro": f1_micro,
               "precision_weighted": precision_weighted,
               "precision_macro": precision_macro,
               "precision_micro": precision_micro,
               "recall_weighted": recall_weighted,
               "recall_macro": recall_macro,
               "recall_micro": recall_micro
               }
    if test:
        results["y_true"] = y_true
        results["y_predicted"] = y_pred_label

    return results


class MovingAverage(object):
    def __init__(self, window, want_increase=True):
        self.moving_avg = [float('-inf')] if want_increase else [float('inf')]
        self.want_increase = want_increase
        self.results = []
        self.window = window

    def add_to_moving_avg(self, x):
        self.results.append(x)
        if len(self.results) >= self.window:
            next_val = sum(self.results[-self.window:]) / self.window
            self.moving_avg.append(next_val)

    def best_result(self, x):
        if self.want_increase:
            return (x - 1e-7) > max(self.results)
        else:
            return (x + 1e-7) < min(self.results)

    def stop(self):
        if len(self.moving_avg) < 2:
            return False
        if self.want_increase:
            return (self.moving_avg[-1] + 1e-7) < self.moving_avg[-2]
        else:
            return (self.moving_avg[-2] + 1e-7) < self.moving_avg[-1]