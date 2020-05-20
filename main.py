from config import FLAGS, COMET_EXPERIMENT
from load_data import load_data
from train import train
from saver import Saver
from eval import eval
import torch

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
    eval_res = eval(preds, test_data)
    eval_res_model = eval(preds_model, test_data)
    print("Test: {}".format(eval_res))
    print("Test_model:{}".format(eval_res_model))
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.test():
            COMET_EXPERIMENT.log_metrics(eval_res, prefix="from_saved")
            COMET_EXPERIMENT.log_metrics(eval_res_model, prefix="latest_model")


if __name__ == "__main__":
    main()
