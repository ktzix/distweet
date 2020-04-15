import pandas as pd
import numpy as np
import transformers
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer

np.random.seed(42)
test = pd.read_csv("/userhome/student/schneider/test.csv", encoding='utf8')
train = pd.read_csv("/userhome/student/schneider/train.csv")

train.drop(columns=['id', 'keyword', 'location'], inplace=True)
train.head

def normalise_text(text):
    text = text.str.lower()  # lowercase
    text = text.str.replace(r"\#", "")  # replaces hashtags
    text = text.str.replace(r"http\S+", "URL")  # remove URL addresses
    text = text.str.replace(r"@", "")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

train["text"] = normalise_text(train["text"])
train.head

train.drop_duplicates(subset="text", inplace = True)

from sklearn.model_selection import train_test_split
train, valid = train_test_split(train, test_size=0.1)

len(train)/len(valid)

# bert tokenizer and bert model loading
from transformers import BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# tokenize, pad and convert texts from tweets to PyTorch tensors
def tokenize_text(df):
    return [
        tokenizer.encode(text, add_special_tokens=True) for text in df.text.values
    ]


def pad_text(tokenized_text):
    max_length = len(max(tokenized_text, key=len))
    return np.array([el + [0] * (max_length-len(el)) for el in tokenized_text])


def tokenize_and_pad_text(df):
    tokenized_text = tokenize_text(df)
    padded_text = pad_text(tokenized_text)
    return torch.tensor(padded_text)


def targets_to_tensor(df):
    return torch.tensor(df.target.values, dtype=torch.long)

#tokenize, pad and convert comments to Pytorch tensors, then use BERT tp transfrom text to embeddings
train_indices = tokenize_and_pad_text(train)
val_indices = tokenize_and_pad_text(valid)
test_indices = tokenize_and_pad_text(test)

with torch.no_grad():
    train_x = bert_model(train_indices)[0]
    val_x = bert_model(val_indices)[0]
    test_x = bert_model(test_indices)[0]

train_y = targets_to_tensor(train)
val_y = targets_to_tensor(valid)
# test_y = targets_to_tensor(test)


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
        super(SimpleClassifier, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        out = self.output_layer(h)
        return out


model = SimpleClassifier(
    input_dim=train_x.size(1),
    output_dim=10,
    hidden_dim=50
)
model

batch_size = 1000
train_iter = BatchedIterator(train_x, train_y, batch_size)
dev_iter = BatchedIterator(val_x, val_y, batch_size)
all_train_loss = []
all_val_loss = []
all_train_acc = []
all_val_acc = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters())

n_epochs = 10
for epoch in range(n_epochs):

    for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
        y_out = model(batch_x)
        loss = criterion(y_out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_out = model(train_x)
    train_loss = criterion(train_out, train_y)
    all_train_loss.append(train_loss.item())
    train_pred = train_out.max(axis=1)[1]
    train_acc = torch.eq(train_pred, train_y).sum().float() / len(train_x)
    all_train_acc.append(train_acc)

    val_out = model(val_x)
    val_loss = criterion(val_out, val_y)
    all_val_loss.append(val_loss.item())
    val_pred = val_out.max(axis=1)[1]
    val_acc = torch.eq(val_pred, val_y).sum().float() / len(val_x)
    all_val_acc.append(val_acc)

    print("Epoch:" + str(epoch) + " \n train accuracy: " + str(train_acc.item()) + " train loss: " + str(
        train_loss.item()))
    print("  val accuracy: " + str(val_acc.item()) + "  dev loss: " + str(val_loss.item()))