import numpy as np
from sklearn import metrics


def eval(preds, dataset):
    y_true = dataset.label_inds[dataset.node_ids]
    y_pred_label = [np.argmax(pred) for pred in preds]
    accuracy = metrics.accuracy_score(y_true, y_pred_label)
    f1 = metrics.f1_score(y_true, y_pred_label, average='weighted')
    results = {"accuracy": accuracy, "f1": f1}
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

    def stop(self):
        if len(self.moving_avg) < 2:
            return False
        if self.want_increase:
            return (self.moving_avg[-1] + 1e-7) < self.moving_avg[-2]
        else:
            return (self.moving_avg[-2] + 1e-7) < self.moving_avg[-1]