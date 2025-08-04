# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# %%
from datasets import load_dataset

# Load the CoNLL-2003 dataset using the 'datasets' library.
dataset = load_dataset('Davlan/conll2003_noMISC', trust_remote_code=True)

dataset

# %%
example = dataset['train'][1000]
for i in zip(example['tokens'], example['ner_tags']):
    print(i)

# %%
from datasets import ClassLabel, Features, Sequence, Value
# Define new features (e.g. for token classification)
new_features = Features({
    'tokens': Sequence(Value("string")),
    'ner_tags': Sequence(
        feature=ClassLabel(
            num_classes=7,
            names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
        )
    )
})

# Cast the dataset to the new schema
dataset = dataset.cast(new_features)

# Accessing the label names from the 'ner_tags' feature.
label_names = dataset['train'].features['ner_tags'].feature.names

label_names
# %%
from transformers import AutoTokenizer

# Define the checkpoint you want to use for the tokenizer.
checkpoint = 'distilbert-base-cased'

# Create a tokenizer instance by loading the pre-trained checkpoint.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# %%
# Tokenize the first training example from the dataset 
token = tokenizer(dataset['train'][0]['tokens'], is_split_into_words = True)

# Print the tokenizer object, the tokenized tokens, and the word IDs
print(token, '\n--------------------------------------------------------------------------------------\n', 
      token.tokens(),'\n--------------------------------------------------------------------------------------\n',
      token.word_ids())
# %%
def align_target(labels, word_ids):
    # Define a mapping from beginning (B-) labels to inside (I-) labels
    begin2inside = {
        1: 2,  # B-LOC -> I-LOC
        3: 4,  # B-MISC -> I-MISC
        5: 6,  # B-ORG -> I-ORG
        7: 8    # B-PER -> I-PER
    }

    # Initialize an empty list to store aligned labels and a variable to track the last word
    align_labels = []
    last_word = None

    # Iterate through the word_ids
    for word in word_ids:
        if word is None:
            label = -100  # Set label to -100 for None word_ids
        elif word != last_word:
            label = labels[word]  # Use the label corresponding to the current word_id
        else:
            label = labels[word]
            # Change B- to I- if the previous word is the same
            if label in begin2inside:
                label = begin2inside[label]  # Map B- to I-

        # Append the label to the align_labels list and update last_word
        align_labels.append(label)
        last_word = word

    return align_labels


# Extract labels and word_ids
labels = dataset['train'][0]['ner_tags']
word_ids = token.word_ids()

# Use the align_target function to align labels
aligned_target = align_target(labels, word_ids)

# Print tokenized tokens, original labels, and aligned labels
for sub_token, lbl, tgt in zip(token.tokens(), labels, aligned_target):
    if tgt != -100:
        print(f"{sub_token} \t\t {lbl} \t {label_names[tgt]} \t ({tgt})")
    else:
        print(f"{sub_token} \t\t {lbl} \t {None} \t ({tgt})")


# %%
def tokenize_fn(batch):
    # Tokenize the input batch
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True)

    # Extract the labels batch from the input batch
    labels_batch = batch['ner_tags']

    # Initialize a list to store aligned targets for each example in the batch
    aligned_targets_batch = []

    # Iterate through each example and align the labels
    for i, labels in enumerate(labels_batch):
        # Extract the word_ids for the current example
        word_ids = tokenized_inputs.word_ids(i)

        # Use the align_target function to align the labels
        aligned_targets_batch.append(align_target(labels, word_ids))

    # Add the aligned labels to the tokenized inputs under the key "labels"
    tokenized_inputs["labels"] = aligned_targets_batch

    # Return the tokenized inputs, including aligned labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)

tokenized_dataset
# %%
tokenized_dataset['train'][0]
# %%
from transformers import DataCollatorForTokenClassification

# Create a DataCollatorForTokenClassification object
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Testing data using the data collator
batch = data_collator([tokenized_dataset['train'][i] for i in range(2)])

# Display the resulting batch
batch
# %%
# Import the seqeval metric from Hugging Face's datasets library
from evaluate import load, list_evaluation_modules

# Load the seqeval metric which can evaluate NER and other sequence tasks
metric = load("seqeval")

# Example usage: compute metric on a sample predictions and reference list 
# predictions and references should be a list of lists containing predicted and true token labels

# List of List Input  
metric.compute(predictions = [['O' , 'B-ORG' , 'I-ORG']], 
               references = [['O' , 'B-MISC' , 'I-ORG']])
# %%
# Function to compute evaluation metrics from model logits and true labels
def compute_metrics(logits_and_labels):

  # Unpack the logits and labels
  logits, labels = logits_and_labels 
  
  # Get predictions from the logits
  predictions = np.argmax(logits, axis=-1)

  # Remove ignored index (special tokens)
  str_labels = [
    [label_names[t] for t in label if t!=-100] for label in labels
  ]
  
  str_preds = [
    [label_names[p] for (p, t) in zip(prediction, label) if t != -100]
    for prediction, label in zip(predictions, labels)
  ]

  # Compute metrics
  results = metric.compute(predictions=str_preds, references=str_labels)
  
  # Extract key metrics
  return {
    "precision": results["overall_precision"],
    "recall": results["overall_recall"], 
    "f1": results["overall_f1"],
    "accuracy": results["overall_accuracy"]  
  }

# %%
# Create mapping from label ID to label string name
id2label = {k: v for k, v in enumerate(label_names)} 

# Create reverse mapping from label name to label ID
label2id = {v: k for k, v in enumerate(label_names)}

print(id2label , '\n--------------------\n' , label2id)
# %%
# Load pretrained token classification model from Transformers 
from transformers import AutoModelForTokenClassification

# Initialize model object with pretrained weights
model = AutoModelForTokenClassification.from_pretrained(
  checkpoint,

  # Pass in label mappings
  id2label=id2label,  
  label2id=label2id
)
model
# %%
# Configure training arguments using TrainigArguments class
from transformers import TrainingArguments

training_args = TrainingArguments(
  # Location to save fine-tuned model 
  output_dir = "ft_ner_kaggle",

  # Evaluate each epoch
  eval_strategy = "epoch",

  # Learning rate for Adam optimizer
  learning_rate = 2e-5, 
  
  # Batch sizes for training and evaluation
  per_device_train_batch_size = 16,
  per_device_eval_batch_size = 16,

  # Number of training epochs
  num_train_epochs = 10,

  # L2 weight decay regularization
  weight_decay = 0.01
)
training_args
# %%
# Initialize Trainer object for model training
from transformers import Trainer

trainer = Trainer(
  # Model to train
  model=model, 
  
  # Training arguments
  args=training_args,

  # Training and validation datasets
  train_dataset=tokenized_dataset["train"],
  eval_dataset=tokenized_dataset["validation"],

  # Tokenizer
  tokenizer=tokenizer,

  # Custom metric function
  compute_metrics=compute_metrics,

  # Data collator
  data_collator=data_collator 
)
trainer
# %%
trainer.train()

# %%
