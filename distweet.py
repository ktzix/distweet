import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
from torch.optim import optimizer
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer

np.random.seed(42)

test = pd.read_csv("/home/schneider/PycharmProjects/mimi/input/nlp/test.csv")
train = pd.read_csv("/home/schneider/PycharmProjects/mimi/input/nlp/train.csv")

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

# bert tokenizer and bert model loading
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')

target_column = "target"
max_seq = 10


# tokenize, pad and convert texts from tweets to PyTorch tensors
def tokenize_text(df, max_seq):
    return [
        tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.text.values
    ]


def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])


def tokenize_and_pad_text(df, max_seq):
    tokenized_text = tokenize_text(df, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)


def targets_to_tensor(df, target_column):
    return torch.tensor(df[target_column].values, dtype=torch.float32)


train_indices = tokenize_and_pad_text(train, max_seq)
test_indices = tokenize_and_pad_text(test, max_seq)

with torch.no_grad():
    train_x = bert_model(train_indices)[0]
    test_x = bert_model(test_indices)[0]

train_y = targets_to_tensor(train, target_column)
test_y = targets_to_tensor(test, target_column)

train_x[0]
train_y[0]
test_x[0]
test_y[0]


class BatchedIterator:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def iterate_once(self):
        for start in range(0, len(self.X), self.batch_size):
            end = start + self.batch_size
            yield self.X[start:end], self.y[start:end]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        out = self.output_layer(h)
        return out


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters())

test_pred = bert_model(test_x).max(axis=1)[1]
test_acc = torch.eq(test_pred, test_y).sum().float() / len(test_x)
test_acc

batch_size = 1000
train_iter = BatchedIterator(train_x, train_y, batch_size)
all_train_loss = []
all_train_acc = []
all_test_loss = []
all_test_acc = []

n_epochs = 10
for epoch in range(n_epochs):

    for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
        y_out = bert_model(batch_x)
        loss = criterion(y_out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_out = bert_model(train_x)
    train_loss = criterion(train_out, train_y)
    all_train_loss.append(train_loss.item())
    train_pred = train_out.max(axis=1)[1]
    train_acc = torch.eq(train_pred, train_y).sum().float() / len(test_x)
    all_train_acc.append(train_acc)

    print(f"Epoch: {epoch}\n  train accuracy: {train_acc}  train loss: {train_loss}")

test_pred = bert_model(test_x).max(axis=1)[1]
test_acc = torch.eq(test_pred, test_y).sum().float() / len(test_x)
test_acc
