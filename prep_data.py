from collections import defaultdict
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import get_data_path
from os.path import join, exists
import pandas as pd
import re

hash_tag_words = ['hashtag_eastasia', 'hashtag', 'hashtag_eastasia_virus', 'hashtag_virus', 'hashtag_virus_othercountry']


def clean_data(dataset):
    clean_text_path = join(get_data_path(), 'corpus', dataset + '_sentences_clean.txt')
    if not exists(clean_text_path):
        docs_list = []
        with open(join(get_data_path(), 'corpus', dataset + '_sentences.txt')) as f:
            for line in f.readlines():
                docs_list.append(line.strip())
        word_counts = defaultdict(int)
        for doc in docs_list:
            temp = clean_doc(doc, dataset)
            words = temp.split()
            for word in words:
                word_counts[word] += 1
        clean_docs = clean_documents(docs_list, word_counts, dataset)
        clean_text_df = pd.DataFrame(clean_docs)
        clean_text_df.to_csv(clean_text_path, index=False)
    clean_text_df = pd.read_csv(clean_text_path)
    split_clean_text = clean_text_df.iloc[:, 0].apply(func=lambda x: len(x.split()))
    print("sentence length statistics")
    print(split_clean_text.describe())


def clean_documents(docs, word_counts, dataset):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print(stop_words)
    ret = []
    for doc in docs:
        doc = clean_doc(doc, dataset)
        words = doc.split()
        words = [word for word in words if word not in stop_words and word_counts[word] >= 5]
        doc = ' '.join(words).strip()
        if doc != '':
            ret.append(' '.join(words).strip())
        else:
            ret.append(' ')
    return ret


def clean_doc_ap(string):
    string = re.sub(r"[^A-Za-z0-9()_+,!?\'\`]", " ", string)  # replace all non alpha numeric characters
    string = re.sub(r"(?<!HASHTAG)_", " ", string)
    string = re.sub(r"(?<!EASTASIA)\+ | (?<!VIRUS)\+", " ", string)
    string = re.sub(r"\+", "_", string)
    string = re.sub(r"HASHTAG_EASTASIA_VIRUS(?!(\s))", "HASHTAG_EASTASIA_VIRUS ", string)
    string = re.sub(r"HASHTAG_EASTASIA(?!(\s|_))", "HASHTAG_EASTASIA ", string)
    string = re.sub(r"HASHTAG_VIRUS(?!(\s|_))", "HASHTAG_VIRUS ", string)
    string = re.sub(r"HASHTAG_VIRUS_OTHERCOUNTRY(?!(\s))", "HASHTAG_VIRUS_OTHERCOUNTRY ", string)
    string = re.sub(r"HASHTAG(?!([\s|_]))", "HASHTAG ", string)
    return string

def clean_doc(string, dataset):
    if dataset == 'twitter_asian_prejudice':
        string = clean_doc_ap(string)
    else:
        raise NotImplementedError
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    dataset = 'twitter_asian_prejudice'
    clean_data(dataset)