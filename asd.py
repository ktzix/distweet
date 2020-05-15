import pandas as pd
import numpy as np
import transformers
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torchtext import data
import spacy
from torch import autograd

train = pd.read_csv("/userhome/student/schneider/train.csv")

train.drop(columns=['id', 'keyword', 'location'], inplace=True)
def normalise_text(text):
    text = text.str.lower()  # lowercase
    text = text.str.replace(r"\#", "")  # replaces hashtags
    text = text.str.replace(r"http\S+", "URL")  # remove URL addresses
    text = text.str.replace(r"@", "")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text
    
train["text"] = normalise_text(train["text"])
train.drop_duplicates(subset="text", inplace = True)
SEED = 1234

torch.manual_seed(SEED)

TEXT = data.Field(tokenize = 'spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)

train.to_csv("train_formatted.csv", index=False)

fields = [('text',TEXT), ('label',LABEL)]


train = data.TabularDataset(
                                        path = 'train_formatted.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header =True
)

import random

train, valid = train.split(split_ratio=0.9, random_state = random.seed(SEED))
TEXT.build_vocab(train, vectors ="glove.6B.100d") 

LABEL.build_vocab(train)



BATCH_SIZE = 64

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True)


import torch.nn as nn
from torch import autograd

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        
        super(LSTMClassifier, self).__init__()
        
        #input: # of words, embeddingdim: # of dimensions for representing
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        
        
        self.fc = nn.Linear(hidden_dim*2 , output_dim)
        
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        
        #text = [batch_size, sent_lenght]
        embedded = self.embedding(text)
        #embedded = [batch_size, sent_lenght, emb_dim]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch_size, numlayers*num diretions, hid dim]
        #cel =[batch size, num layers* num directions, hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        
        #hidden = [batch_size, hid_dim*num directions]
        dense_outputs=self.fc(hidden)
        
        outputs=self.act(dense_outputs)
        
        return outputs
        
   INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 7
OUTPUT_DIM = 1
num_layers =2
bidirectional = True
dropout = 0.2

model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers, bidirectional, dropout)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.BCELoss()

from sklearn.metrics import classification_report
def class_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc
    
    
    from sklearn.metrics import accuracy_score
import torch.nn.functional as F
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        
        #reset gradients after every batch
        optimizer.zero_grad()
        
        
        texts, text_lengths = batch.text
        
            #convert to 1D tensor
        predictions = model(texts, text_lengths).squeeze()
        #predictions = model(batch.text)
        

        loss = criterion(predictions, batch.label)
        
        acc = class_accuracy(predictions, batch.label)
        
        
        loss.backward()
        
        #update the weights
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    #deactivates dropout layers
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            texts, text_lengths = batch.text
        
            #convert to 1D tensor
            predictions = model(texts, text_lengths).squeeze()
            #predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = class_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    val_loss, val_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    #save the best model
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print("Epoch:"  + str(epoch) + " Time" + str(epoch_mins) + str(epoch_secs) )
    print (" \n train accuracy: " + str(train_acc) + " train loss: " + str(train_loss) )
    print("  val accuracy: " + str(val_acc) + "  val loss: " + str(val_loss) )
