
from config import FLAGS, COMET_EXPERIMENT
import torch
from eval import eval
from load_data import load_data
from saver import Saver
from train import train


def main():
    saver = Saver()
    train_data, val_data, test_data = load_data()
    print(train_data.graph.shape)
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.train():
            saved_model, model = train(train_data, val_data, test_data, saver)
    else:
        saved_model, model = train(train_data, val_data, test_data, saver)
    with torch.no_grad():
        test_loss_model, preds_model = model(train_data.get_pyg_graph(device=FLAGS.device), test_data)
    eval_res = eval(preds_model, test_data, True)
    print("Test: {}".format(eval_res))
    if COMET_EXPERIMENT:
        from comet_ml.utils import ConfusionMatrix
        y_true = eval_res.pop('y_true')
        y_pred = eval_res.pop('y_pred')

        def index_to_example(index):
            test_docs_ids = test_data.node_ids
            return test_data.docs[test_docs_ids[index]]

        confusion_matrix = ConfusionMatrix(index_to_example_function=index_to_example)
        confusion_matrix.compute_matrix(y_true, y_pred)

        with COMET_EXPERIMENT.test():
            COMET_EXPERIMENT.log_metrics(eval_res)
            COMET_EXPERIMENT.log_confusion_matrix(matrix=confusion_matrix, labels=list(test_data.label_dict.keys()))



if __name__ == "__main__":
    main()
