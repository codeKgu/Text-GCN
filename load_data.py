from config import FLAGS
from os.path import join
from utils import get_save_path, load, save
from build_graph import build_text_graph_dataset
from dataset import TextDataset
import gc


def load_data():
    dir = join(get_save_path(), 'split')
    dataset_name = FLAGS.dataset
    train_ratio = int(FLAGS.tvt_ratio[0] * 100)
    val_ratio = int(FLAGS.tvt_ratio[1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    save_fn = '{}_train_{}_val_{}_test_{}_seed_{}_window_size_{}'.format(dataset_name, train_ratio,
                                                          val_ratio, test_ratio,
                                                          FLAGS.random_seed, FLAGS.word_window_size)
    path = join(dir, save_fn)
    rtn = load(path)
    if rtn:
        train_data, val_data, test_data = rtn['train_data'], rtn['val_data'], rtn['test_data']
    else:
        train_data, val_data, test_data = _load_tvt_data_helper()
        save({'train_data': train_data, 'val_data': val_data, 'test_data': test_data}, path)
    return train_data, val_data, test_data


def _load_tvt_data_helper():
    dir = join(get_save_path(), 'all')
    path = join(dir, FLAGS.dataset + '_all_window_' + str(FLAGS.word_window_size))
    rtn = load(path)
    if rtn:
        dataset = TextDataset(None, None, None, None, None, None, rtn)
    else:
        dataset = build_text_graph_dataset(FLAGS.dataset, FLAGS.word_window_size)
        gc.collect()
        save(dataset.__dict__, path)

    train_dataset, val_dataset, test_dataset = dataset.tvt_split(FLAGS.tvt_ratio[:2], FLAGS.tvt_list, FLAGS.random_seed)
    return train_dataset, val_dataset, test_dataset