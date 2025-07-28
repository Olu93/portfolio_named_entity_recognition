from transformers import DistilBertForTokenClassification
import torch.nn as nn
from transformers import AutoModelForTokenClassification

class WeightedDistilBertForTokenClassification(DistilBertForTokenClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_state = outputs[0]        # (bs, seq_len, hidden_size)
        logits = self.classifier(hidden_state)  # (bs, seq_len, num_labels)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits
        }



class WeightedTokenClassifier(AutoModelForTokenClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # don't use default loss
            **kwargs
        )
        logits = outputs.logits  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits
        }


def tokenize_and_align_labels_no_subtokens(texts, labels, tokenizer, label2id):
    """Tokenize texts and align labels with subword tokens"""
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    aligned_labels = []
    
    try:
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD) get -100 label
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Only label the first token of a given word
                    if word_idx < len(label):
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        # Handle truncation case
                        label_ids.append(-100)
                else:
                    # Subtokens get -100 label
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            aligned_labels.append(label_ids)
    except Exception as e:
        print(f"Error processing sentence {i}: {e}")
        print(f"Sentence: {texts[i]}")
        print(f"Labels: {labels[i]}")
        print(f"Label length: {len(labels[i])}")
        print(f"Word IDs: {word_ids}")
        print(f"Max word_idx: {max(word_ids) if word_ids else 'N/A'}")
        raise e
    
    return tokenized_inputs, aligned_labels


def tokenize_and_align_labels(texts, labels, tokenizer, label2id):
    """Tokenize texts and align labels with subword tokens"""
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    aligned_labels = []
    
    try:
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD) get -100 label
                    label_ids.append(-100)
                elif word_idx < len(label):  # Check bounds
                    if word_idx != previous_word_idx:
                        # New word - use the original label
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        # Same word, continuation - use the same label but convert B- to I-
                        original_label = label[word_idx]
                        if original_label.startswith('B-'):
                            # Convert B- to I- for continuation tokens
                            continuation_label = 'I-' + original_label[2:]
                            label_ids.append(label2id[continuation_label])
                        else:
                            # Keep I- or O labels as is
                            label_ids.append(label2id[original_label])
                else:
                    # Handle case where tokenizer creates more tokens than we have labels
                    # This can happen with truncation or special tokenization
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            aligned_labels.append(label_ids)
    except Exception as e:
        print(f"Error processing sentence {i}: {e}")
        print(f"Sentence: {texts[i]}")
        print(f"Labels: {labels[i]}")
        print(f"Label length: {len(labels[i])}")
        print(f"Word IDs: {word_ids}")
        print(f"Max word_idx: {max(word_ids) if word_ids else 'N/A'}")
        raise e
    
    return tokenized_inputs, aligned_labels