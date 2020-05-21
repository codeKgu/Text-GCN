import networkx as nx
import scipy.sparse as sp
from math import log
from dataset import TextDataset
from collections import defaultdict
import pandas as pd
from utils import get_corpus_path
from os.path import join, exists
from tqdm import tqdm


def build_text_graph_dataset(dataset, window_size):
    if "small" in dataset or "presplit" in dataset:
        dataset_name = "_".join(dataset.split("_")[:-1])
    else:
        dataset_name = dataset
    clean_text_path = join(get_corpus_path(), dataset_name + '_sentences_clean.txt')
    labels_path = join(get_corpus_path(), dataset_name + '_labels.txt')
    labels = pd.read_csv(labels_path, header=None, sep='\t')
    doc_list = []
    f = open(clean_text_path, 'rb')
    for line in f.readlines():
        doc_list.append(line.strip().decode())
    f.close()
    assert len(labels) == len(doc_list)
    if 'presplit' not in dataset:
        labels_list = labels.iloc[0:, 0].tolist()
        split_dict = None
    else:
        labels_list = labels.iloc[0:, 2].tolist()
        split = labels.iloc[0:, 1].tolist()
        split_dict = {}
        for i, v in enumerate(split):
            split_dict[i] = v
    if "small" in dataset:
        doc_list = doc_list[:200]
        labels_list = labels_list[:200]

    word_freq = get_vocab(doc_list)
    vocab = list(word_freq.keys())
    if not exists(join(get_corpus_path(), dataset + '_vocab.txt')):
        vocab_str = '\n'.join(vocab)
        f = open(join(get_corpus_path(), dataset + '_vocab.txt'), 'w')
        f.write(vocab_str)
        f.close()
    words_in_docs, word_doc_freq = build_word_doc_edges(doc_list)
    word_id_map = {word: i for i, word in enumerate(vocab)}

    sparse_graph = build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size)
    docs_dict = {i: doc for i, doc in enumerate(doc_list)}
    return TextDataset(dataset, sparse_graph, labels_list, vocab, word_id_map, docs_dict, None,
                       train_test_split=split_dict)


def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    doc_word_freq = defaultdict(int) # frequency of document word pair
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj


def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
            word_freq[word] += 1
    return word_freq


def build_word_doc_edges(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq


if __name__ == "__main__":
    dataset = 'twitter_asian_prejudice'
    build_text_graph_dataset(dataset, 20)