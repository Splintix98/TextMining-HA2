import torch
import torch.nn as nn
from utils import dev

class RNN_model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0, device=dev)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, num_layers=10) # input_dimension, hidden_dimension
        # self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=10, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.vocab_size, device=dev)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1, device=dev)
    
    def forward(self, sentence, inference):
        sentence = sentence.reshape(-1)
        inference = inference.reshape(-1)
        h0_initial_random = torch.rand(10, self.hidden_size).to(dev)
        def copy_forward(self, s):            
            inp = self.emb(s)
            h_0 = h0_initial_random
            all_hidden_states, last_hidden_state = self.rnn(inp, h_0)
            return all_hidden_states, last_hidden_state
        _, last_hidden_sentence = copy_forward(self, sentence)
        _, last_hidden_inference = copy_forward(self, inference)
        similarity = torch.cosine_similarity(last_hidden_sentence, last_hidden_inference, dim=0)
        #similarity = torch.bmm(last_hidden_sentence, last_hidden_inference.T)
        #output = torch.sigmoid(similarity)        
        # output = (similarity + 1) * 0.5 # normalize to [0, 1] while it originally is [-1, 1]
        output = self.fc(similarity)
        return output