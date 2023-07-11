import pickle
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from dataclasses import dataclass
from collections import Counter
import string
import nltk
import torch
nltk.download('stopwords')


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vocab = Counter()
word2index = {}
index2word = {}
max_len = 0


def get_data(fname):
    # import file with pickle
    with open(fname, 'rb') as f:
        # data = [(['w', 'w', ...], [1, 2, 3, ...]), ([], []), (), ...]
        data = pickle.load(f)

    for tuple in data:
        add_sent_to_vocab(normalize_sent(tuple[0]))
    
    set_max_len(data)
    return data


def normalize_sent(sent):
    normalized_sentence = [word.lower() for word in sent]
    return normalized_sentence


# adds words of current sentence to vocab
def add_sent_to_vocab(sent):
    for w in sent:
        vocab[w] += 1


def build_word2index():
    i = 1  # index 0 will be kept for padding
    for w in vocab:
        # if vocab[w] > 5:
        word2index[w] = i
        index2word[i] = w
        i += 1


def set_max_len(data):
    global max_len
    for entry in data:
        if len(entry[0]) > max_len:
            max_len = len(entry[0])


class Data(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index][0]
        labels = self.data[index][1]

        sentence = [word2index[w.lower()] for w in sentence]

        # necessary when using batching
        # if len(sentence)>max_len:
        #     sentence = sentence[:max_len]
        # else:
        #     sentence.extend([0 for _ in range(max_len-len(sentence))])

        # if len(labels)>max_len:
        #     labels = labels[:max_len]
        # else:
        #     labels.extend([0 for _ in range(max_len-len(labels))])

        sentence_tnsr = torch.LongTensor(sentence).to(dev)
        labels_tnsr = torch.LongTensor(labels).to(dev)
        
        return sentence_tnsr, labels_tnsr
