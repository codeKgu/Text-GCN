from eval import eval, MovingAverage
from config import FLAGS, COMET_EXPERIMENT
from model_factory import create_model
import torch
import time


def train(train_data, val_data, test_data, saver):
    train_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    val_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    model = create_model(train_data)
    model = model.to(FLAGS.device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number params: ", pytorch_total_params)
    moving_avg = MovingAverage(FLAGS.validation_window_size, FLAGS.validation_metric != 'loss')
    pyg_graph = train_data.get_pyg_graph(FLAGS.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, )

    for epoch in range(FLAGS.num_epochs):
        t = time.time()
        model.train()
        model.zero_grad()
        loss, preds_train = model(pyg_graph, train_data)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if COMET_EXPERIMENT:
            COMET_EXPERIMENT.log_metric("loss", loss, epoch + 1)
        with torch.no_grad():
            val_loss, preds_val = model(pyg_graph, val_data)
            val_loss = val_loss.item()
            eval_res_val = eval(preds_val, val_data)
            print("Epoch: {:04d}, Train Loss: {:.5f}, Time: {:.5f}".format(epoch, loss, time.time() - t))
            print("Val Loss: {:.5f}".format(val_loss))
            print("Val Results: {}\n\n".format(eval_res_val))
            eval_res_val["loss"] = val_loss
            if COMET_EXPERIMENT:
                COMET_EXPERIMENT.log_metrics(eval_res_val, prefix="validation", step=epoch+1)

            if len(moving_avg.results) == 0 or moving_avg.best_result(eval_res_val[FLAGS.validation_metric]):
                saver.save_trained_model(model, epoch + 1)
            moving_avg.add_to_moving_avg(eval_res_val[FLAGS.validation_metric])
            if moving_avg.stop():
                break
    best_model = saver.load_trained_model(train_data)
    return best_model, model