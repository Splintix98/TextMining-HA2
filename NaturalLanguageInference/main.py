import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
import nltk
import pickle
import utils
import os
import utils
from utils import vocab, Data
from model import RNN_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm


if __name__ == '__main__':
    train_data = utils.get_data('NaturalLanguageInference/WNLI/train.tsv')
    val_data = utils.get_data('NaturalLanguageInference/WNLI/dev.tsv')
    test_data = utils.get_data('NaturalLanguageInference/WNLI/test.tsv')   
    
    utils.build_word2index()
    
    vocab_size = len(utils.vocab)
    emb_dim = 300
    hidden_size = 256
    model = RNN_model(vocab_size, emb_dim, hidden_size).to(utils.dev)
    
    # Prepare datasets and dataloaders
    train_dataset = Data(train_data)
    val_dataset = Data(val_data)
    test_dataset = Data(test_data)    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    
    def validate(lm_model, val_loader):
        cls_pred = []
        cls_original = []
        with torch.no_grad():
            for sent, inf, label in val_loader:
                out = lm_model(sent, inf)
                out = torch.round(out)
                cls_pred.extend(out.reshape(-1).cpu().numpy().tolist())
                cls_original.extend(label.reshape(-1).cpu().numpy().tolist())
        
        return accuracy_score(cls_original, cls_pred) 
    
    #Training
    def train(lm_model, train_loader, val_loader):
        lm_model.train()
        val_accuracy = 0
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(lm_model.parameters(), lr=0.001)
        epochs = 10
        for e in range(epochs):
            for sent, inf, label in tqdm(train_loader):
                out = lm_model(sent, inf)
                out = torch.round(out)
                #print(out)
                #print(y.reshape(-1))
                loss = criterion(out, label.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            lm_model.eval()
            acc = validate(lm_model, val_loader)
            print(f"accuracy at the end of epoch - {e}: {acc}")
            if acc > val_accuracy:
                torch.save(lm_model.state_dict(), 'inference_best.pt')
            
            lm_model.train()

    train(model, train_loader, val_loader)
    
    
    
