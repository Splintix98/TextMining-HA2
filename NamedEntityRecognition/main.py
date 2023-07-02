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

def validate(lm_model, val_loader):
        lm_model.eval()
        cls_pred = []
        cls_original = []
        with torch.no_grad():
            for sent, inf, label in val_loader:
                out = lm_model(sent, inf)
                out = torch.argmax(out, dim=1)
                cls_pred.extend(out.reshape(-1).cpu().numpy().tolist())
                cls_original.extend(label.reshape(-1).cpu().numpy().tolist())
        
        return accuracy_score(cls_original, cls_pred) 

#Training
def train(lm_model, train_loader, val_loader, epochs):
    lm_model.train()
    max_val_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(lm_model.parameters(), lr=0.0001)
    for e in range(epochs):
        for sent, labels in tqdm(train_loader):
            sent = sent.squeeze(0)            
            out = lm_model(sent)    
            loss = criterion(out, labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc = validate(lm_model, val_loader)
        print(f"accuracy at the end of epoch - {e}: {acc}")
        if acc > max_val_accuracy:
            torch.save(lm_model.state_dict(), 'inference_best.pt')
            max_val_accuracy = acc
        
        lm_model.train()
    return max_val_accuracy

if __name__ == '__main__':
    train_data = utils.get_data('NamedEntityRecognition/conll2003_train.pkl')
    val_data = utils.get_data('NamedEntityRecognition/conll2003_val.pkl')  
    test_data = utils.get_data('NamedEntityRecognition/conll2003_test.pkl')     
    
    utils.build_word2index()
    
    vocab_size = len(utils.vocab)
    max_len = utils.max_len
    emb_dim = 100
    hidden_size = 128
    batch_size = 1
    num_layers = 3
    model = RNN_model(vocab_size, emb_dim, hidden_size, max_len, batch_size, num_layers).to(utils.dev)
    
    # Prepare datasets and dataloaders
    train_dataset = Data(train_data)
    val_dataset = Data(val_data)
    test_dataset = Data(test_data)    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    
    max_val_accuracy = train(model, train_loader, val_loader, epochs=200)
    
    print(f"max_val_accuracy: {max_val_accuracy}")
    with open('inference_best_params.pkl', 'wb') as f:
        pickle.dump([utils.word2index, vocab_size, emb_dim, hidden_size, batch_size, num_layers, max_val_accuracy], f)


    
    
    
