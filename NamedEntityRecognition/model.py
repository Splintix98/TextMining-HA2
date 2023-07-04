import nltk.corpus
import torch
import torch.nn as nn
from utils import dev
import torch.nn.init as init


class RNN_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, max_len, batch_size=32, num_layers=12, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.emb = nn.Embedding(self.vocab_size+1, self.emb_dim, padding_idx=0, device=dev)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, self.num_layers) # input_dimension, hidden_dimension
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size, 9, device=dev)
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
    
    def forward(self, sentence):
        inp = self.emb(sentence)
        h_0 = torch.rand(self.num_layers, self.hidden_size).to(dev)
        all_hidden_states, _ = self.rnn(inp, h_0)

        logits = self.fc1(all_hidden_states)
        logits = self.relu(logits)
        log_probs = torch.log_softmax(logits, dim=1) #e.g. [10,9] 10 words, 9 classes.
        #Can be passed as is to NLLLoss criterion.
        #log_softmax is used to avoid underflow/overflow.

        return log_probs