from dataclasses import dataclass
from collections import Counter
import string
import nltk
import torch
# nltk.download('stopwords')
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset
# import torchtext
import numpy as np

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

    def clean_word(word):
        w = word.lower()
        if w in puncts:
            return None
        if w in s_words:
            return None
        w = w.translate({ord(c): None for c in puncts}) # remove single punctuations from word
        
        return w
        

    normalized_sentence = [clean_word(word) for word in sent if clean_word(word) is not None]
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


class WNLIData(Dataset):
    def __init__(self, data:list[WNLI_Item]): #, tokenizer: torchtext.legacy.data.utils.Tokenizer):
        self.data = data
        # self.tokenizer = tokenizer
        self.sentences = [wnli_item.sentence for wnli_item in data]
        self.inferences = [wnli_item.inference for wnli_item in data]
        self.labels = [wnli_item.label for wnli_item in data]

        self.sent_features = None # = np.zeros((len(self.data), 200), dtype=np.int64)
        self.inf_features = None # = np.zeros((len(self.data), 200), dtype=np.int64)

        if(len(self.sentences) != len(self.inferences) or len(self.sentences) != len(self.labels)):
            raise Exception("Lengths of sentences, inferences and labels are not equal!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # sentence = self.sentences[index]
        # inference = self.inferences[index]
        # label = self.labels[index]
        # return sentence, inference, label
    
        # sentence = torch.tensor(self.sentences[index])
        # inference = torch.tensor(self.inferences[index])
        # label = torch.tensor(self.labels[index])
        # return sentence, inference, label
    
        # sentence = self.tokenizer(self.sentences[index])
        # inference = self.tokenizer(self.inferences[index])
        # label = torch.tensor(self.labels[index])
        # return sentence, inference, label

        sentence_tnsr = torch.LongTensor(self.sent_features[index]).to(dev)
        inference_tnsr = torch.LongTensor(self.inf_features[index]).to(dev)
        label_tnsr = torch.FloatTensor([int(self.data[index].label)]).to(dev)

        # print(sentence_tnsr.shape, inference_tnsr.shape, label_tnsr.shape) # --> torch.Size([31]) torch.Size([31]) torch.Size([1])

        return sentence_tnsr.reshape(-1), inference_tnsr.reshape(-1), label_tnsr.reshape(-1)
    
    
    # returns length of longest sentence or inference in this dataset
    def get_longest_sent_len(self):
        max_sent_len = max([len(wnli_item.sentence) for wnli_item in self.data])
        max_inf_len = max([len(wnli_item.inference) for wnli_item in self.data])
        return max(max_sent_len, max_inf_len)
    
    # returns an array of ints representing the sentence by taking the word2indices
    def convert_sent_to_ints(self, sent):
        return [word2index[w] for w in sent]
    
    def convert_inf_to_ints(self, inf):
        return [word2index[w] for w in inf]
    
    # builds features for sentences and inferences
    #   a feature is an array of ints representing the sentece or inference
    #   the array is padded with 0s to a length uniform across all three datasets (test, train, val)
    def build_features(self, length=200):
        self.sent_features = np.zeros((len(self.sentences), length), dtype=np.int64)
        self.inf_features = np.zeros((len(self.inferences), length), dtype=np.int64)

        for i, sentence in enumerate(self.sentences):
            sent_ints = self.convert_sent_to_ints(sentence)
            self.sent_features[i, -len(sent_ints):] = np.array(sent_ints)[:length]

        for i, inference in enumerate(self.inferences):
            inf_ints = self.convert_sent_to_ints(inference)
            self.inf_features[i, -len(inf_ints):] = np.array(inf_ints)[:length]
        
        # print("sentences[0]: ", self.sentences[0])
        # print("features shape: ", self.sent_features.shape)
        # print("features[0]: ", self.sent_features[0])
    
    