import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch.nn.init as init

import n_utils as utils


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        print(f"Input size: {input_size}, Hidden size: {hidden_size}, Num classes: {num_classes}")
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)

    def forward(self, sentence, inference):
        sentence_emb = self.embedding(sentence)
        inference_emb = self.embedding(inference)

        sent_out, _ = self.lstm(sentence_emb)
        inf_out, _ = self.lstm(inference_emb)

        sent_final_hidden = sent_out[:, -1, :]
        inf_final_hidden = inf_out[:, -1, :]

        combined_hidden = torch.cat((sent_final_hidden, inf_final_hidden), dim=1)

        output = self.fc(combined_hidden)
        # print(output)
        return output


if __name__ == '__main__':
    train_data = utils.get_data('WNLI/train.tsv')
    val_data = utils.get_data('WNLI/dev.tsv')
    test_data = utils.get_data('WNLI/test.tsv')

    # print(f"Train data size: {len(train_data)}")
    # print(f"TrainData[1]: {train_data[1]}")

    utils.build_word2index()
    vocab_size = len(utils.vocab)

    train_dataset = utils.WNLIData(train_data)
    val_dataset = utils.WNLIData(val_data)
    test_dataset = utils.WNLIData(test_data)

    max_sent_len = max(train_dataset.get_longest_sent_len(), val_dataset.get_longest_sent_len(), test_dataset.get_longest_sent_len())
    print(f"Max sent len: {max_sent_len}")

    train_dataset.build_features(max_sent_len)
    val_dataset.build_features(max_sent_len)
    test_dataset.build_features(max_sent_len)

    """"""
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel(vocab_size, 256, 1).to(device)

    criterion = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for sent, inf, label in train_loader:
            sent = sent.to(device)
            inf = inf.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(sent, inf)
            loss = criterion(outputs, label)
            # print(f"Loss: {loss}")
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for sent, inf, label in val_loader:
                sent = sent.to(device)
                inf = inf.to(device)
                label = label.to(device)

                outputs = model(sent, inf)
                _, predicted = torch.max(outputs.data, dim=1)
                total_samples += label.size(0)
                total_correct += (predicted == label).sum().item()
            accuracy = total_correct / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")
    """"""
