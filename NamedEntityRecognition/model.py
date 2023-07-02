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
        # self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=10, batch_first=True)
        # self.lin = nn.Linear(self.hidden_size, self.vocab_size, device=dev)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size, 9, device=dev)
        self.softmax = nn.Softmax(dim=0)
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
    
    def forward(self, sentence):
        #batch_size = sentence.shape[0]
        inp = self.emb(sentence)
        #inp = inp.permute(1, 0, 2)
        h_0 = torch.rand(self.num_layers, self.hidden_size).to(dev)
        all_hidden_states, last_hidden_state = self.rnn(inp, h_0)
        #all_hidden_states_perm = all_hidden_states.permute(1, 0, 2)
        #sliced_tensor = all_hidden_states_perm[:, -128:].unsqueeze(-2)

        probabilities = []
        for word in all_hidden_states:
            word = self.fc1(word)
            word = self.relu(word)
            word = self.softmax(word)
            probabilities.append(torch.argmax(word, dim=0))
    
        output = torch.stack(probabilities)      
        return output 