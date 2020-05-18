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
            trained_model = train(train_data, val_data, saver)
    else:
        trained_model = train(train_data, val_data, saver)
    with torch.no_grad():
        trained_model.eval()
        test_loss, preds = trained_model(train_data.get_pyg_graph(device=FLAGS.device), test_data)

    eval_res = eval(preds, test_data)
    print("Test: {}".format(eval_res))
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.test():
            COMET_EXPERIMENT.log_metrics(eval_res)


if __name__ == "__main__":
    main()
