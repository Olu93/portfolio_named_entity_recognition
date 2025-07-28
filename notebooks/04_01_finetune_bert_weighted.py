# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoConfig
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from notebooks import FILES_DIR
from transformers import EarlyStoppingCallback
import torch.nn as nn



# Try to import seqeval for better evaluation metrics
try:
    from seqeval.metrics import precision_score, recall_score, f1_score
    SEQEVAL_AVAILABLE = True
    print("✓ seqeval available for advanced NER metrics")
except ImportError:
    print("⚠ seqeval not available. Install with: pip install seqeval")
    print("   Falling back to simple accuracy metrics")
    SEQEVAL_AVAILABLE = False

# %%
# Define the label mapping for BIO tags
label2id = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8
}

id2label = {v: k for k, v in label2id.items()}

print("Label mapping:")
for label, id_ in label2id.items():
    print(f"  {label}: {id_}")

# %%
# Load and parse the CoNLL file
def load_conll_data(file_path):
    """Load data from CoNLL format file"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            line = line.strip()
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    if tag not in label2id:
                        print(f"About to do something stupid at line {idx}. The tag is ({tag})")
                    current_sentence.append((token, tag))
    
    # Add the last sentence if it exists
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

# Load the CoNLL data
conll_file = FILES_DIR / "ner_annotations.conll"
sentences = load_conll_data(conll_file)

print(f"Loaded {len(sentences)} sentences from CoNLL file")

# Show some examples
print("\nExample sentences:")
for i, sentence in enumerate(sentences[:2]):
    print(f"\nSentence {i+1}:")
    for token, tag in sentence[:10]:  # Show first 10 tokens
        print(f"  {token} -> {tag}")
    if len(sentence) > 10:
        print("  ...")

# %%
# Prepare data for tokenization
def prepare_data_for_tokenization(sentences):
    """Convert sentences to format suitable for tokenization"""
    texts = []
    labels = []
    
    for sentence in sentences:
        tokens = [token for token, _ in sentence]
        tags = [tag for _, tag in sentence]
        
        # Join tokens with spaces for tokenization
        text = " ".join(tokens)
        texts.append(text)
        labels.append(tags)
    
    return texts, labels

# %%
# Prepare the data
texts, labels = prepare_data_for_tokenization(sentences)

print(f"Prepared {len(texts)} texts for tokenization")

# %%
# Initialize tokenizer - use cased for better NER performance
# model_name = "bert-base-cased"  # Changed from uncased for better NER
model_name = "distilbert-base-cased"  # Changed from uncased for better NER
tokenizer = AutoTokenizer.from_pretrained(model_name)
from notebooks.weighted_models import WeightedTokenClassifier as TokenClassifier

print(f"Using tokenizer: {model_name}")

# %%
from notebooks.weighted_models import tokenize_and_align_labels as tokenize_and_align_labels

# Tokenize the data
print("Tokenizing and aligning labels...")
tokenized_inputs, aligned_labels = tokenize_and_align_labels(texts, labels, tokenizer, label2id)

print(f"Tokenized {len(tokenized_inputs['input_ids'])} sequences")

# %%
# Create dataset
class NERDataset(TorchDataset):
    def __init__(self, tokenized_inputs, labels):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# %%
# Split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Train set: {len(train_texts)} samples")
print(f"Validation set: {len(val_labels)} samples")

# %%
# Tokenize train and validation sets
print("Tokenizing training data...")
train_tokenized, train_aligned_labels = tokenize_and_align_labels(
    train_texts, train_labels, tokenizer, label2id
)

print("Tokenizing validation data...")
val_tokenized, val_aligned_labels = tokenize_and_align_labels(
    val_texts, val_labels, tokenizer, label2id
)

# %%
# Create datasets
train_dataset = NERDataset(train_tokenized, train_aligned_labels)
val_dataset = NERDataset(val_tokenized, val_aligned_labels)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")



# %%
# Initialize model with proper configuration
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,  # if using BERT
    attention_probs_dropout_prob=0.3
)

model = TokenClassifier.from_pretrained(
    model_name,
    config=config
)

print(f"Model initialized with {len(label2id)} labels")
print(f"Model config: {model.config}")

# %%
# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# %%
# Compute metrics function for evaluation
def compute_metrics(pred):
    """Compute precision, recall, and F1 score for NER"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    
    if SEQEVAL_AVAILABLE:
        precision = precision_score(true_labels, true_predictions, average="weighted")
        recall = recall_score(true_labels, true_predictions, average="weighted")
        f1 = f1_score(true_labels, true_predictions, average="weighted")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        # Fallback to simple accuracy
        correct = 0
        total = 0
        for true_pred, true_label in zip(true_predictions, true_labels):
            for p, l in zip(true_pred, true_label):
                if p == l:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

# %%
# Training arguments with advanced optimizations
total_steps = len(train_dataset) // 16 * 3  # Approximate total steps
warmup_steps = int(0.1 * total_steps)  # 10% warmup

training_args = TrainingArguments(
    output_dir="./bert-ner-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    warmup_steps=warmup_steps,  # Warmup for stable training
    logging_dir="./logs",
    logging_steps=10,  # More frequent logging
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1" if SEQEVAL_AVAILABLE else "eval_loss",
    greater_is_better=True if SEQEVAL_AVAILABLE else False,
    push_to_hub=False,
    fp16=True,  # Mixed precision training for speed and memory
    gradient_accumulation_steps=2,  # Effective batch size = 16 * 2 = 32
    dataloader_pin_memory=True,  # Faster data loading
    remove_unused_columns=False,  # Keep all columns for evaluation
    report_to=None,  # Disable wandb/tensorboard reporting
)

print(f"Training arguments:")
print(f"  Total steps: ~{total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Mixed precision: {training_args.fp16}")
print(f"  Best metric: {training_args.metric_for_best_model}")

# %%
# Initialize trainer with metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# %%
# Train the model
print("Starting training...")
trainer.train()


# %%
# Evaluate on validation set
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Validation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# %%
# Test the model on a few examples
def predict_entities(text, model, tokenizer, id2label):
    """Predict entities in a given text"""
    # Tokenize the text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(model.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert predictions to labels
    predicted_labels = [id2label[label_id.item()] for label_id in predictions[0]]
    
    # Align with tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    aligned_predictions = []
    
    for token, label in zip(tokens, predicted_labels):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            aligned_predictions.append((token, label))
    
    return aligned_predictions


# Test examples
test_texts = [
    "Joe Biden visited the White House in Washington DC.",
    "Apple CEO Tim Cook announced new products at the conference.",
    "The United Nations met in New York to discuss climate change.",
    "A drug is available for monkeypox patients who have or who are at risk of severe disease, but doctors say they continue to face challenges getting access to it. The US Food and Drug Administration hasn’t approved tecovirimat – sold under the brand name Tpoxx – specifically for use against monkeypox, but the US Centers for Disease Control and Prevention has made the drug available from the Strategic National Stockpile through expanded access during the global outbreak that has caused about 5,800 probable or confirmed cases in the US.",
    "Tpoxx was FDA-approved in 2018 as the first drug to treat smallpox, a virus in the same family as monkeypox. The World Health Organization declared smallpox eradicated in 1980, but concerns that the virus could be weaponized drove the US government to stockpile more than 1.7 million courses of the drug in case of a bioterrorism event. ",
    "Tpoxx is approved in the European Union to treat monkeypox as well as smallpox. It can be taken intravenously or more commonly as an oral pill. Tpoxx is considered experimental when it comes to monkeypox treatment because there’s no data to prove its effectiveness against the disease in humans. Its safety was assessed in healthy humans before its FDA approval for smallpox, and its effectiveness has been tested in animals infected with viruses related to smallpox, including monkeypox. ",
    "As the ongoing outbreak increases demand for the drug, the FDA and CDC recently eased some of the administrative requirements that health care providers face when requesting access. However, doctors across the country suggest that significant barriers remain, causing some patients to wait days for shipments or travel to find medical centers that can provide the product at all. “Patients are trying hard to get this medication, even going out of city or out of state in some cases,” said Dr. Peter Chin-Hong, an infectious disease physician at UCSF Health."
]

print("Testing model on example texts:")
for text in test_texts:
    print(f"\nText: {text}")
    predictions = predict_entities(text, model, tokenizer, id2label)
    print("Predictions:")
    for token, label in predictions:
        # if label != 'O':
        print(f"  {token} -> {label}")

# %%
# Save the model and tokenizer properly
model_save_path = FILES_DIR / "bert_ner_finetuned"
trainer.save_model(str(model_save_path))
tokenizer.save_pretrained(str(model_save_path))

print(f"Model and tokenizer saved to {model_save_path}")




# %%
# Save comprehensive training results
results = {
    "model_name": model_name,
    "num_labels": len(label2id),
    "label_mapping": label2id,
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "eval_results": eval_results,  # Save all metrics
    "model_save_path": str(model_save_path),
    "training_config": {
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
        "fp16": training_args.fp16,
    }
}

with open(FILES_DIR / "bert_training_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("Training results saved to bert_training_results.json")

# %%
# Optional: Test with different BERT variants
print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Model saved to: {model_save_path}")
print(f"Results saved to: {FILES_DIR / 'bert_training_results.json'}")
print("\nTo use the model for inference:")
print(f"from transformers import AutoTokenizer, AutoModelForTokenClassification")
print(f"tokenizer = AutoTokenizer.from_pretrained('{model_save_path}')")
print(f"model = AutoModelForTokenClassification.from_pretrained('{model_save_path}')")

# %%
