import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
import nltk
import pickle
import utils
import os




if __name__ == '__main__':
    train_data = utils.get_data('WNLI/train.tsv')
    val_data = utils.get_data('WNLI/dev.tsv')
    test_data = utils.get_data('WNLI/test.tsv')    
    
    
