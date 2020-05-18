from eval import eval, MovingAverage
from config import FLAGS, COMET_EXPERIMENT
from model_factory import create_model
import torch


def train(train_data, val_data, saver):
    train_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    val_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    model = create_model(train_data)
    model = model.to(FLAGS.device)
    moving_avg = MovingAverage(FLAGS.validation_window_size, FLAGS.validation_metric != 'loss')
    pyg_graph = train_data.get_pyg_graph(FLAGS.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, )

    for epoch in range(FLAGS.num_epochs):
        model.train()
        model.zero_grad()
        loss, _ = model(pyg_graph, train_data)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print("Epoch: {}, Train Loss: {}".format(epoch, loss))
        if COMET_EXPERIMENT:
            COMET_EXPERIMENT.log_metric("loss", loss, epoch + 1)
        with torch.no_grad():
            model = model.to(FLAGS.device)
            model.eval()
            val_loss, preds = model(pyg_graph, val_data)
            eval_res = eval(preds, val_data)
            if epoch % FLAGS.print_every_epochs == 0:

                print("Val Loss: {:.7f}".format(val_loss))
                print("Val Results: {}\n\n".format(eval_res))
            eval_res["loss"] = val_loss
            if COMET_EXPERIMENT:
                COMET_EXPERIMENT.log_metrics(eval_res, prefix="validation", step=epoch+1)

            if len(moving_avg.results) == 0 or (eval_res[FLAGS.validation_metric] - 1e-7) > max(moving_avg.results):
                saver.save_trained_model(model, epoch + 1)
            moving_avg.add_to_moving_avg(eval_res[FLAGS.validation_metric])
            if moving_avg.stop():
                break
    best_model = saver.load_trained_model(train_data)
    return best_model