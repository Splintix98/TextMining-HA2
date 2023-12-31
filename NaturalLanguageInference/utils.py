from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from dataclasses import dataclass
from collections import Counter
import string
import nltk
import torch
nltk.download('stopwords')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class WNLI_Item:
    id: int
    sentence: str
    inference: str
    label: int


vocab = Counter()
word2index = {}
index2word = {}


def get_data(fname):
    data: list[WNLI_Item] = []
    with open(fname) as fs:
        lines_after_first = fs.readlines()[1:]
        for line in lines_after_first:
            try:
                id, sent_1, sent_2, label = line.strip().split('\t')
            except:
                print(line)
                continue

            normalized_sent_1 = normalize_sent(sent_1.split())
            normalized_sent_2 = normalize_sent(sent_2.split())

            add_sent_to_vocab(normalized_sent_1)
            add_sent_to_vocab(normalized_sent_2)

            data.append(WNLI_Item(id, normalized_sent_1, normalized_sent_2, label))
    
    return data


def normalize_sent(sent: list[str]):
    s_words = stopwords.words('english')
    puncts = string.punctuation

    normalized_sentence = [word.lower().translate(str.maketrans('', '', string.punctuation)) for word in sent if word.lower() not in s_words]
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


class Data(Dataset):
    def __init__(self, data):
        self.data = data
        self.sentences = [d.sentence for d in data]
        self.inferences = [d.inference for d in data]
        self.labels = [d.label for d in data]
        self.max_len = max([len(s) for s in self.sentences])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = [word2index[w] for w in self.data[index].sentence]
        inference = [word2index[w] for w in self.data[index].inference]
        label = self.data[index].label

        if len(sentence) > self.max_len:
            sentence = sentence[:self.max_len]
        else:
            sentence.extend([0 for _ in range(self.max_len-len(sentence))])

        if len(inference) > self.max_len:
            inference = inference[:self.max_len]
        else:
            inference.extend([0 for _ in range(self.max_len-len(inference))])

        sentence_tnsr = torch.LongTensor(sentence).to(dev)
        inference_tnsr = torch.LongTensor(inference).to(dev)
        label_tnsr = torch.LongTensor([int(self.data[index].label)]).to(dev)
        
        return sentence_tnsr, inference_tnsr, label_tnsr
