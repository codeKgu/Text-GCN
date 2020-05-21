
from config import FLAGS, COMET_EXPERIMENT
import torch
from eval import eval
from load_data import load_data
from saver import Saver
from train import train


def main():
    saver = Saver()
    train_data, val_data, test_data = load_data()
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.train():
            saved_model, model = train(train_data, val_data, test_data, saver)
    else:
        saved_model, model = train(train_data, val_data, test_data, saver)
    with torch.no_grad():
        test_loss, preds = saved_model(train_data.get_pyg_graph(device=FLAGS.device), test_data)
        test_loss_model, preds_model = model(train_data.get_pyg_graph(device=FLAGS.device), test_data)
    eval_res = eval(preds, test_data, True)
    eval_res_model = eval(preds_model, test_data, True)
    print("Test: {}".format(eval_res))
    print("Test_model:{}".format(eval_res_model))
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.test():
            c_mat = eval_res.pop('confusion_matrix')
            c_mat1 = eval_res_model.pop('confusion_matrix')
            COMET_EXPERIMENT.log_metrics(eval_res, prefix="from_saved")
            COMET_EXPERIMENT.log_metrics(eval_res_model, prefix="latest_model")
            print(test_data.label_dict)
            COMET_EXPERIMENT.log_confusion_matrix(matrix=c_mat, labels=list(test_data.label_dict.keys()))
            COMET_EXPERIMENT.log_confusion_matrix(matrix=c_mat1, labels=list(test_data.label_dict.keys()))


if __name__ == "__main__":
    main()
