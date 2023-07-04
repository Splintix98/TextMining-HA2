from model import RNN_model
import pickle
import utils
import torch
from main import validate
from torch.utils.data import DataLoader
import numpy as np


with open('NER_best_params.pkl', 'rb') as f:
    vocab, vocab_size, emb_dim, hidden_size, max_len, batch_size, num_layers, max_val_accuracy = pickle.load(f)  

utils.word2index = vocab
test_data = utils.get_data('NamedEntityRecognition/conll2003_test.pkl')  
test_dataset = utils.Data(test_data)    
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
      

acc = []
runs = 20
for i in range(runs):
    best_model = RNN_model(vocab_size, emb_dim, hidden_size, max_len, batch_size, num_layers).to(utils.dev)
    best_model.load_state_dict(torch.load('NER_best.pt'))
    acc.append(validate(best_model, test_loader))
print(f"Mean accuracy on test set for {runs} runs: {np.mean(acc)}")