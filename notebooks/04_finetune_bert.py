# %% [markdown]
# # BERT Fine-tuning for Named Entity Recognition (NER)
# 
# This notebook implements fine-tuning of a pre-trained BERT model for Named Entity Recognition using the CoNLL-2003 format dataset. We'll use the DistilBERT model with custom loss functions and advanced training optimizations to achieve optimal NER performance.
# 
# ## Setup: Import required libraries and dependencies

# %%
from notebook_config import FILES_DIR, DATASETS_DIR
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoConfig
)
from sklearn.model_selection import train_test_split
import json
from transformers import EarlyStoppingCallback
from notebooks.notebook_finetune_utils import tokenize_and_align_labels as tokenize_and_align_labels
from notebooks.notebook_finetune_utils import DistilBertWithHingeLoss as TokenClassifier
# from transformers import AutoModelForTokenClassification as TokenClassifier

# Try to import seqeval for better evaluation metrics
try:
    from seqeval.metrics import precision_score, recall_score, f1_score
    SEQEVAL_AVAILABLE = True
    print("✓ seqeval available for advanced NER metrics")
except ImportError:
    print("⚠ seqeval not available. Install with: pip install seqeval")
    print("   Falling back to simple accuracy metrics")
    SEQEVAL_AVAILABLE = False

# %% [markdown]
# ## Define BIO Tag Label Mapping
# 
# Create the label mapping for BIO (Begin-Inside-Outside) tagging scheme:
# 
# **BIO Tag Structure:**
# - **B-**: Beginning of an entity
# - **I-**: Inside/continuation of an entity
# - **O**: Outside/not part of any entity
# 
# **Entity Types:**
# - **PER**: Person names
# - **ORG**: Organization names
# - **LOC**: Location names
# - **MISC**: Miscellaneous entities
# 
# This mapping ensures proper conversion between string labels and numeric IDs for model training.

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

# %% [markdown]
# ## Load and Parse CoNLL Format Data
# 
# Load the NER annotations from the CoNLL-2003 format file created in the previous notebook:
# 
# **CoNLL Format Structure:**
# - Each line contains: `token TAB label`
# - Empty lines separate sentences
# - Tokens are pre-tokenized and aligned with their entity labels
# 
# **Data Validation:**
# - Checks for valid BIO tags
# - Handles malformed lines gracefully
# - Provides detailed error reporting for debugging

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
conll_file = DATASETS_DIR / "ner_annotations_combined.conll"
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

# %% [markdown]
# ## Prepare Data for Tokenization
# 
# Convert the CoNLL format data into a format suitable for BERT tokenization:
# 
# **Data Transformation:**
# - **Token extraction**: Separate tokens from their labels
# - **Text reconstruction**: Join tokens with spaces for BERT tokenization
# - **Label preservation**: Maintain corresponding label sequences
# 
# This step prepares the data for the tokenization process while preserving the entity annotations.

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

# %% [markdown]
# ## Initialize BERT Tokenizer
# 
# Set up the BERT tokenizer for text processing:
# 
# **Model Selection:**
# - **dslim/distilbert-NER**: Pre-trained on NER tasks, optimized for entity recognition
# - **Cased tokenizer**: Preserves case information important for NER
# - **Specialized performance**: Better than generic BERT for NER tasks
# 
# **Tokenizer Features:**
# - Handles subword tokenization
# - Manages special tokens (CLS, SEP, PAD)
# - Provides token-to-word mapping for label alignment

# %%
# Initialize tokenizer - use cased for better NER performance
# model_name = "bert-base-cased"  # Changed from uncased for better NER
model_name = "dslim/distilbert-NER"  # Changed from uncased for better NER
tokenizer = AutoTokenizer.from_pretrained(model_name)
# from notebooks.notebook_finetune_utils import DistilBertWithHingeLoss as TokenClassifier

print(f"Using tokenizer: {model_name}")

# %% [markdown]
# ## Tokenize and Align Labels
# 
# Apply BERT tokenization to the text data and align the BIO labels with the tokenized output:
# 
# **Tokenization Process:**
# - **Subword splitting**: BERT breaks words into subword units
# - **Label alignment**: Maps original labels to tokenized sequences
# - **Special token handling**: Properly handles CLS, SEP, and PAD tokens
# - **Truncation and padding**: Ensures consistent sequence lengths
# 
# This step is crucial for maintaining the relationship between tokens and their entity labels.

# %%
# Tokenize the data
print("Tokenizing and aligning labels...")
tokenized_inputs, aligned_labels = tokenize_and_align_labels(texts, labels, tokenizer, label2id)

print(f"Tokenized {len(tokenized_inputs['input_ids'])} sequences")

# %% [markdown]
# ## Create PyTorch Dataset Class
# 
# Define a custom PyTorch Dataset class for efficient data loading during training:
# 
# **Dataset Features:**
# - **Efficient indexing**: Fast access to individual samples
# - **Tensor conversion**: Automatic conversion of labels to PyTorch tensors
# - **Memory optimization**: Loads data on-demand rather than all at once
# - **Compatibility**: Works seamlessly with PyTorch DataLoader
# 
# This class provides the interface between our processed data and the training loop.

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

# %% [markdown]
# ## Split Data into Training and Validation Sets
# 
# Divide the dataset into training and validation subsets for proper model evaluation:
# 
# **Split Strategy:**
# - **80% training, 20% validation**: Standard split ratio for NER tasks
# - **Random state**: Ensures reproducible splits across runs
# - **Stratified sampling**: Maintains entity distribution across splits
# - **Independent evaluation**: Prevents data leakage between train and validation
# 
# This separation is essential for unbiased model performance assessment.

# %%
# Split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Train set: {len(train_texts)} samples")
print(f"Validation set: {len(val_labels)} samples")

# %% [markdown]
# ## Tokenize Training and Validation Sets Separately
# 
# Apply tokenization to the split datasets to prepare them for model training:
# 
# **Processing Steps:**
# - **Training set tokenization**: Creates tokenized inputs for model training
# - **Validation set tokenization**: Prepares data for evaluation during training
# - **Label alignment**: Ensures proper alignment for both datasets
# - **Consistent processing**: Same tokenization applied to both splits
# 
# This step ensures both datasets are properly formatted for the training pipeline.

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

# %% [markdown]
# ## Initialize BERT Model with Custom Configuration
# 
# Set up the BERT model with optimized configuration for NER tasks:
# 
# **Model Configuration:**
# - **Pre-trained base**: Uses dslim/distilbert-NER as starting point
# - **Label count**: Configured for our 9 BIO tag classes
# - **Dropout settings**: Optimized for regularization and generalization
# - **Custom loss**: Uses hinge loss for better NER performance
# 
# **Advanced Features:**
# - **Attention dropout**: Reduces overfitting in attention mechanisms
# - **Layer dropout**: Regularizes transformer layers
# - **Classifier dropout**: Prevents overfitting in the final classification layer

# %%
# Initialize model with proper configuration
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    # Dropout settings (DistilBERT)
    dropout=0.3,
    attention_dropout=0.1,

    # Bonus (ignored by DistilBERT but OK to include for compatibility)
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.1,
    classifier_dropout=0.3,
    summary_first_dropout=0.3,
    layerdrop=0.1,
)

model = TokenClassifier.from_pretrained(
    model_name,
    config=config
)

print(f"Model initialized with {len(label2id)} labels")
print(f"Model config: {model.config}")

# %% [markdown]
# ## Set Up Data Collator for Batch Processing
# 
# Configure the data collator for efficient batch processing during training:
# 
# **Data Collator Functions:**
# - **Dynamic padding**: Pads sequences to the longest in each batch
# - **Label handling**: Properly handles label padding and alignment
# - **Memory efficiency**: Optimizes memory usage during training
# - **Batch consistency**: Ensures uniform batch structure
# 
# This component is essential for efficient training with variable-length sequences.

# %%
# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# %% [markdown]
# ## Define Evaluation Metrics Function
# 
# Create a comprehensive evaluation function that computes NER-specific metrics:
# 
# **Evaluation Metrics:**
# - **Precision**: Accuracy of positive predictions
# - **Recall**: Coverage of actual entities
# - **F1 Score**: Harmonic mean of precision and recall
# - **Entity-level evaluation**: Considers complete entity spans
# 
# **Fallback Handling:**
# - Uses seqeval for advanced NER metrics when available
# - Falls back to simple accuracy when seqeval is not installed
# - Handles special tokens (-100) properly during evaluation

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

# %% [markdown]
# ## Configure Advanced Training Arguments
# 
# Set up comprehensive training configuration with optimizations for NER tasks:
# 
# **Training Optimizations:**
# - **Learning rate**: 2e-5 for stable fine-tuning
# - **Batch size**: 5 per device with gradient accumulation for effective batch size of 32
# - **Mixed precision**: FP16 for faster training and reduced memory usage
# - **Warmup steps**: 10% of total steps for stable training start
# - **Early stopping**: Prevents overfitting with patience of 2 epochs
# 
# **Advanced Features:**
# - **Gradient accumulation**: Simulates larger batch sizes
# - **Model checkpointing**: Saves best model based on validation metrics
# - **Logging**: Comprehensive training progress tracking

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
    metric_for_best_model="precision" if SEQEVAL_AVAILABLE else "eval_loss",
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

# %% [markdown]
# ## Initialize Trainer with Callbacks
# 
# Set up the Hugging Face Trainer with all necessary components:
# 
# **Trainer Components:**
# - **Model**: Our configured BERT model
# - **Datasets**: Training and validation datasets
# - **Tokenizer**: For text processing during evaluation
# - **Data collator**: For batch processing
# - **Metrics function**: For comprehensive evaluation
# - **Callbacks**: Early stopping to prevent overfitting
# 
# This trainer provides a complete training pipeline with built-in evaluation and checkpointing.

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# %% [markdown]
# ## Execute Model Training
# 
# Start the fine-tuning process with comprehensive monitoring:
# 
# **Training Process:**
# - **Epoch-based training**: 30 epochs with early stopping
# - **Validation evaluation**: After each epoch
# - **Progress tracking**: Detailed logging of loss and metrics
# - **Model checkpointing**: Saves best model based on validation performance
# 
# **Monitoring Features:**
# - Real-time loss tracking
# - Validation metric updates
# - **Early stopping**: Prevents overfitting
# - **Best model preservation**: Keeps the best performing model

# %%
# Train the model
print("Starting training...")
train_results = trainer.train()
train_results

# %% [markdown]
# ## Evaluate Model Performance
# 
# Assess the trained model's performance on the validation set:
# 
# **Evaluation Metrics:**
# - **Precision**: How many predicted entities are correct
# - **Recall**: How many actual entities were found
# - **F1 Score**: Balanced measure of precision and recall
# - **Loss**: Training and validation loss comparison
# 
# **Performance Analysis:**
# - Identifies model strengths and weaknesses
# - Guides potential model improvements
# - Validates training effectiveness

# %%
# Evaluate on validation set
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Validation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# %% [markdown]
# ## Test Model on Example Texts
# 
# Create a prediction function and test the model on various example texts:
# 
# **Prediction Function Features:**
# - **Text tokenization**: Proper BERT tokenization with truncation
# - **Entity prediction**: Token-level entity classification
# - **Label conversion**: Maps numeric predictions back to BIO tags
# - **Token alignment**: Handles subword tokenization properly
# 
# **Test Examples:**
# - Simple entity examples
# - Complex medical text
# - Multi-entity sentences
# - Edge cases for validation

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
    "A drug is available for monkeypox patients who have or who are at risk of severe disease, but doctors say they continue to face challenges getting access to it. The US Food and Drug Administration hasn't approved tecovirimat – sold under the brand name Tpoxx – specifically for use against monkeypox, but the US Centers for Disease Control and Prevention has made the drug available from the Strategic National Stockpile through expanded access during the global outbreak that has caused about 5,800 probable or confirmed cases in the US.",
    "Tpoxx was FDA-approved in 2018 as the first drug to treat smallpox, a virus in the same family as monkeypox. The World Health Organization declared smallpox eradicated in 1980, but concerns that the virus could be weaponized drove the US government to stockpile more than 1.7 million courses of the drug in case of a bioterrorism event. ",
    "Tpoxx is approved in the European Union to treat monkeypox as well as smallpox. It can be taken intravenously or more commonly as an oral pill. Tpoxx is considered experimental when it comes to monkeypox treatment because there's no data to prove its effectiveness against the disease in humans. Its safety was assessed in healthy humans before its FDA approval for smallpox, and its effectiveness has been tested in animals infected with viruses related to smallpox, including monkeypox. ",
    "As the ongoing outbreak increases demand for the drug, the FDA and CDC recently eased some of the administrative requirements that health care providers face when requesting access. However, doctors across the country suggest that significant barriers remain, causing some patients to wait days for shipments or travel to find medical centers that can provide the product at all. \"Patients are trying hard to get this medication, even going out of city or out of state in some cases,\" said Dr. Peter Chin-Hong, an infectious disease physician at UCSF Health."
]

print("Testing model on example texts:")
for text in test_texts:
    print(f"\nText: {text}")
    predictions = predict_entities(text, model, tokenizer, id2label)
    print("Predictions:")
    for token, label in predictions:
        # if label != 'O':
        print(f"  {token} -> {label}")

# %% [markdown]
# ## Save Trained Model and Tokenizer
# 
# Persist the trained model and tokenizer for future use:
# 
# **Model Persistence:**
# - **Model weights**: Complete trained model parameters
# - **Tokenizer**: Vocabulary and tokenization rules
# - **Configuration**: Model architecture and settings
# - **Metadata**: Training information and results
# 
# **Storage Benefits:**
# - Enables model reuse without retraining
# - Supports deployment in production systems
# - **Version control**: Tracks model iterations
# - **Sharing**: Allows model distribution

# %%
# Save the model and tokenizer properly
model_save_path = FILES_DIR / "pretrained" / "dslim_bert_ner_finetuned"
trainer.save_model(str(model_save_path))
tokenizer.save_pretrained(str(model_save_path))

print(f"Model and tokenizer saved to {model_save_path}")

# %% [markdown]
# ## Save Comprehensive Training Results
# 
# Export detailed training results and configuration for analysis and reproducibility:
# 
# **Results Documentation:**
# - **Model configuration**: Architecture and hyperparameters
# - **Training metrics**: Performance statistics and evaluation results
# - **Dataset information**: Training and validation set sizes
# - **Training configuration**: Learning rate, batch size, epochs, etc.
# 
# **Analysis Benefits:**
# - **Reproducibility**: Complete training setup documentation
# - **Performance tracking**: Historical model performance
# - **Hyperparameter analysis**: Impact of different settings
# - **Model comparison**: Basis for comparing different models

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

model_save_path_results = model_save_path / "results"
model_save_path_results.mkdir(parents=True, exist_ok=True)

with open(model_save_path_results / "bert_training_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("Training results saved to bert_training_results.json")

# %% [markdown]
# ## Training Completion Summary
# 
# Provide a comprehensive summary of the training process and next steps:
# 
# **Training Summary:**
# - **Model location**: Where the trained model is saved
# - **Results location**: Where training metrics are stored
# - **Usage instructions**: How to load and use the model
# - **Performance highlights**: Key achievements and metrics
# 
# **Next Steps:**
# - Model deployment instructions
# - Inference code examples
# - Potential improvements and optimizations

# %%
# Optional: Test with different BERT variants
print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Model saved to: {model_save_path}")
print(f"Results saved to: {model_save_path_results / 'bert_training_results.json'}")
print("\nTo use the model for inference:")
print(f"from transformers import AutoTokenizer, AutoModelForTokenClassification")
print(f"tokenizer = AutoTokenizer.from_pretrained('{model_save_path}')")
print(f"model = AutoModelForTokenClassification.from_pretrained('{model_save_path}')")

# %% [markdown]
# ## Save Training Progress History
# 
# Export the complete training log history for detailed analysis:
# 
# **Training History:**
# - **Step-by-step metrics**: Loss and evaluation metrics at each step
# - **Learning curves**: Training and validation performance over time
# - **Convergence analysis**: How the model learned during training
# - **Debugging information**: Detailed logs for troubleshooting
# 
# **Analysis Benefits:**
# - **Learning curve analysis**: Identify training patterns
# - **Overfitting detection**: Monitor validation vs training performance
# - **Hyperparameter tuning**: Guide future optimization efforts
# - **Model comparison**: Compare different training runs

# %%
# NOTE 520 Left
# save training progress 
# %%
train_log_history = trainer.state.log_history
with open(model_save_path_results / "bert_training_log_history.json", 'w') as f:
    json.dump(train_log_history, f, indent=2)

# %% [markdown]
# ## Final Model Testing
# 
# Perform final validation tests on the trained model to ensure quality and readiness for deployment.

# %%
# test the model

# %% [markdown]
# ## Training Progress Visualization
# 
# Create comprehensive visualizations of the training progress to analyze model convergence and performance metrics over time.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

# %%
# Load training history from JSON file (or use trainer state if available)
try:
    # Try to load from the saved JSON file first
    import json
    log_file_path = model_save_path_results / "bert_training_log_history.json"
    if log_file_path.exists():
        with open(log_file_path, 'r') as f:
            train_log_history = json.load(f)
        print(f"Loaded training history from {log_file_path}")
    else:
        # Fallback to trainer state
        train_log_history = trainer.state.log_history
        print("Using trainer state for training history")
except Exception as e:
    print(f"Error loading training history: {e}")
    train_log_history = []

# Convert to DataFrame
df_history = pd.DataFrame(train_log_history)

print(f"Training history contains {len(df_history)} log entries")
print(f"Columns: {list(df_history.columns)}")

# %%
# Create figure with two subplots for loss and metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Training and Evaluation Loss
if 'loss' in df_history.columns and 'eval_loss' in df_history.columns:
    # Separate training and evaluation data
    train_data = df_history[['step', 'loss']].dropna()
    eval_data = df_history[['step', 'eval_loss']].dropna()
    
    if len(train_data) > 0:
        ax1.plot(train_data['step'], train_data['loss'], label='Training Loss', 
                linewidth=2, color='blue', alpha=0.8)
    
    if len(eval_data) > 0:
        ax1.plot(eval_data['step'], eval_data['eval_loss'], label='Validation Loss', 
                linewidth=2, color='red', alpha=0.8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Progression', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for final values
    if len(train_data) > 0:
        final_train_loss = train_data['loss'].iloc[-1]
        ax1.annotate(f'Final Train Loss: {final_train_loss:.4f}', 
                    xy=(train_data['step'].iloc[-1], final_train_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                    fontsize=10)
    
    if len(eval_data) > 0:
        final_eval_loss = eval_data['eval_loss'].iloc[-1]
        ax1.annotate(f'Final Eval Loss: {final_eval_loss:.4f}', 
                    xy=(eval_data['step'].iloc[-1], final_eval_loss),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                    fontsize=10)
else:
    ax1.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Training and Validation Loss Progression')

# Plot 2: F1, Precision, and Recall Metrics
metrics_columns = ['eval_f1', 'eval_precision', 'eval_recall']
available_metrics = [col for col in metrics_columns if col in df_history.columns]

if available_metrics:
    # Filter out None values and convert to numeric
    df_metrics = df_history[['step'] + available_metrics].copy()
    df_metrics = df_metrics.dropna()
    
    if len(df_metrics) > 0:
        colors = ['green', 'orange', 'purple']
        for i, metric in enumerate(available_metrics):
            metric_name = metric.replace('eval_', '').title()
            ax2.plot(df_metrics['step'], df_metrics[metric], 
                    label=metric_name, linewidth=2, color=colors[i % len(colors)], alpha=0.8)
        
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics Progression (F1, Precision, Recall)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.5)  # Adjust based on your data range
        
        # Add annotations for final values
        for i, metric in enumerate(available_metrics):
            final_value = df_metrics[metric].iloc[-1] if len(df_metrics) > 0 else None
            if final_value is not None:
                metric_name = metric.replace('eval_', '').title()
                ax2.annotate(f'Final {metric_name}: {final_value:.4f}', 
                            xy=(df_metrics['step'].iloc[-1], final_value),
                            xytext=(10, 10 + i*15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                            fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No metrics data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Validation Metrics Progression')
else:
    ax2.text(0.5, 0.5, 'No evaluation metrics available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Validation Metrics Progression')

plt.tight_layout()
plt.show()

# %%
# Save the plots
plot_save_path = model_save_path_results / "training_progression_plots.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"Training progression plots saved to: {plot_save_path}")

# %%
# Create a comprehensive summary table of training progress
print("\n" + "="*80)
print("TRAINING PROGRESS SUMMARY")
print("="*80)

if len(df_history) > 0:
    # Get the last entry to show final metrics
    last_entry = df_history.iloc[-1]
    
    print(f"\n📊 Training Overview:")
    print(f"  Total Log Entries: {len(df_history)}")
    print(f"  Total Training Steps: {df_history['step'].max()}")
    print(f"  Training Epochs: {df_history['epoch'].max():.1f}")
    
    # Show final training metrics
    print(f"\n🎯 Final Training Metrics:")
    if 'train_loss' in df_history.columns:
        print(f"  Final Training Loss: {df_history['train_loss'].iloc[-1]:.6f}")
    if 'train_runtime' in df_history.columns:
        runtime_hours = df_history['train_runtime'].iloc[-1] / 3600
        print(f"  Total Training Time: {runtime_hours:.2f} hours")
    
    # Show final evaluation metrics
    print(f"\n📈 Final Evaluation Metrics:")
    eval_metrics = ['eval_loss', 'eval_f1', 'eval_precision', 'eval_recall']
    for metric in eval_metrics:
        if metric in df_history.columns:
            final_value = df_history[metric].iloc[-1]
            if pd.notna(final_value):
                metric_name = metric.replace('eval_', '').title()
                print(f"  {metric_name}: {final_value:.6f}")
    
    # Show progression analysis
    if len(df_history) > 1:
        print(f"\n📈 Training Progression Analysis:")
        
        # Loss progression
        if 'loss' in df_history.columns:
            train_loss_data = df_history[['step', 'loss']].dropna()
            if len(train_loss_data) > 0:
                initial_loss = train_loss_data['loss'].iloc[0]
                final_loss = train_loss_data['loss'].iloc[-1]
                loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
                
                print(f"  Training Loss:")
                print(f"    Initial: {initial_loss:.6f}")
                print(f"    Final: {final_loss:.6f}")
                print(f"    Improvement: {loss_improvement:.2f}%")
        
        # Evaluation loss progression
        if 'eval_loss' in df_history.columns:
            eval_loss_data = df_history[['step', 'eval_loss']].dropna()
            if len(eval_loss_data) > 0:
                initial_eval_loss = eval_loss_data['eval_loss'].iloc[0]
                final_eval_loss = eval_loss_data['eval_loss'].iloc[-1]
                eval_loss_improvement = ((initial_eval_loss - final_eval_loss) / initial_eval_loss) * 100
                
                print(f"  Validation Loss:")
                print(f"    Initial: {initial_eval_loss:.6f}")
                print(f"    Final: {final_eval_loss:.6f}")
                print(f"    Improvement: {eval_loss_improvement:.2f}%")
        
        # F1 score progression
        if 'eval_f1' in df_history.columns:
            f1_data = df_history[['step', 'eval_f1']].dropna()
            if len(f1_data) > 0:
                initial_f1 = f1_data['eval_f1'].iloc[0]
                final_f1 = f1_data['eval_f1'].iloc[-1]
                f1_improvement = ((final_f1 - initial_f1) / initial_f1) * 100 if initial_f1 > 0 else 0
                
                print(f"  F1 Score:")
                print(f"    Initial: {initial_f1:.6f}")
                print(f"    Final: {final_f1:.6f}")
                print(f"    Improvement: {f1_improvement:.2f}%")
        
        # Learning rate analysis
        if 'learning_rate' in df_history.columns:
            lr_data = df_history[['step', 'learning_rate']].dropna()
            if len(lr_data) > 0:
                initial_lr = lr_data['learning_rate'].iloc[0]
                final_lr = lr_data['learning_rate'].iloc[-1]
                print(f"  Learning Rate:")
                print(f"    Initial: {initial_lr:.2e}")
                print(f"    Final: {final_lr:.2e}")
                print(f"    Range: {lr_data['learning_rate'].min():.2e} - {lr_data['learning_rate'].max():.2e}")

else:
    print("❌ No training history data available")

print("\n" + "="*80)

