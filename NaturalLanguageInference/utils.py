from dataclasses import dataclass
from collections import Counter
import string
import nltk
import torch
nltk.download('stopwords')
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset

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
    data:list[WNLI_Item] = []
    with open(fname) as fs:
        lines_after_first = fs.readlines()[1:]
        for line in lines_after_first:
            try:
                id, sent1, sent2, label = line.strip().split('\t')
            except:
                print(line)
                continue
            normalize_sent1 = normalize_sent(sent1.split())
            normalize_sent2 = normalize_sent(sent2.split())
            add_sent_to_vocab(normalize_sent1)
            add_sent_to_vocab(normalize_sent2)
            data.append(WNLI_Item(id, normalize_sent1, normalize_sent2, label))
    return data
    
def normalize_sent(sent):
    s_words = stopwords.words('english')
    puncts = string.punctuation

    normalized_sentence = [word.lower() for word in sent if word not in puncts and word.lower() not in s_words]
    return normalized_sentence

# adds words of current sentence to vocab
def add_sent_to_vocab(sent):        
    for w in sent:
        vocab[w]+=1    

def build_word2index():
    i = 1 # index 0 will be kept for padding    
    for w in vocab:
        #if vocab[w]>5:
        word2index[w] = i
        index2word[i] = w
        i+=1


class Data(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence_tnsr = torch.LongTensor([word2index[w] for w in self.data[index].sentence]).to(dev)
        inference_tnsr = torch.LongTensor([word2index[w] for w in self.data[index].inference]).to(dev)
        label_tnsr = torch.FloatTensor([int(self.data[index].label)]).to(dev)
        return sentence_tnsr, inference_tnsr, label_tnsr