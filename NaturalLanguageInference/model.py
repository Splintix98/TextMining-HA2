import nltk.corpus
import torch
import torch.nn as nn
from utils import dev
import torch.nn.init as init


class RNN_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, batch_size=32, num_layers=12, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.vocab_size+1, self.emb_dim, padding_idx=0, device=dev)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, self.num_layers) # input_dimension, hidden_dimension
        # self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=10, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.vocab_size, device=dev)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.num_layers*2*hidden_size, hidden_size, device=dev)
        self.fc2 = nn.Linear(hidden_size, batch_size, device=dev)
        self.fc3 = nn.Linear(batch_size, 2, device=dev)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
    
    def forward(self, sentence, inference):
        batch_size = sentence.shape[0]
        def copy_forward(self, s):            
            inp = self.emb(s)
            inp = inp.permute(1, 0, 2)
            h_0 = torch.rand(self.num_layers, batch_size, self.hidden_size).to(dev)
            all_hidden_states, last_hidden_state = self.rnn(inp, h_0)
            return all_hidden_states, last_hidden_state
        sent_out, last_hidden_sentence = copy_forward(self, sentence)
        inf_out, last_hidden_inference = copy_forward(self, inference)    
        concat  = torch.cat((last_hidden_sentence, last_hidden_inference), dim=2)
        concat = concat.reshape(batch_size, -1) #e.g. (32, 12, 512) after concat becomes (32, 6144)
        output = self.fc1(concat)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output