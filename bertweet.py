import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import tensorflow as tf
import torch.optim as optim
import time
import copy
from torch.utils.data import Dataset

import torch
print('There are %d GPU(s) available.' % torch.cuda.device_count())

print('We will use the GPU:', torch.cuda.get_device_name(0))

device = torch.device("cuda")

np.random.seed(42)
train = pd.read_csv("/userhome/student/schneider/train.csv", encoding='UTF-8')

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
train.head
sentences = train.text.values[:6]
labels = train.target.values[:6]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

max_len = 85
all_tokens = []
all_masks = []
all_segments = []

for text in sentences:
    text = tokenizer.tokenize(text)
    
    text = text[:max_len-2]
    input_sek = ["[CLS]"] + text +["[SEP]"]
    pad_len = max_len-len(input_sek)
    
    tokens=tokenizer.convert_tokens_to_ids(input_sek)
    tokens +=[0] * pad_len
    pad_masks = [1] * len(input_sek) + [0] * pad_len
    segment_ids = [0] * max_len
    
    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)
    


print('token_ids: ', all_masks[1])
labels= torch.tensor(labels)
input_ids=torch.tensor(all_tokens)
all_masks=torch.tensor(all_masks)

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, all_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('token_ids: ', train_dataset[1])

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32
# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

#author recommended values
#Batch size: 16, 32
#Learning rate (Adam): 5e-5, 3e-5, 2e-5
#Number of epochs: 2, 3, 4
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, 
                  eps = 1e-8
                )

from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

epochs = 2

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            warmup_steps = 0, # Default value in run_glue.py
                                            t_total = total_steps)
                                           

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
    
    import random
seed_val=42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    t0 = time.time()
    total_train_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
            
        model.zero_grad()
            
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        
        print(" training loss: {0:.2f}".format(loss.item()))    
        total_train_loss += loss.item()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
            
        scheduler.step()
            
    avg_train_loss = total_train_loss / len(train_dataloader)
            
    training_time = format_time(time.time() - t0)
            
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
            
            
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
            
    for batch in validation_dataloader:
        
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
            
        with torch.no_grad():
                
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        print("  val Loss: {0:.2f}".format(loss.item()))
        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
                
        print("  val Loss: {0:.2f}".format(flat_accuracy(logits, label_ids)))
        total_eval_accuracy += flat_accuracy(logits, label_ids)
                
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
                
    validation_time = format_time(time.time() - t0)
                
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
        
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

