# %% [markdown]
# # Data Preparation and Cleaning Pipeline
# 
# This notebook transforms the raw dataset into a clean, structured format suitable for model fine-tuning. We'll address data quality issues, normalize entity formats, and prepare the data for both training and evaluation phases.
# 
# ## Setup environment and import necessary libraries

# %%
from notebook_config import DATASETS_DIR
import pandas as pd
import matplotlib.pyplot as plt
from markdownify import markdownify as md

# %% [markdown]
# ## Load and Preview Raw Dataset
# 
# Load the full dataset into a DataFrame and preview its structure to understand the current state of the data before applying any transformations. This establishes our baseline for measuring the impact of our cleaning operations.

# %%
df = pd.read_csv(DATASETS_DIR / 'full_data.csv')
df.head()

# %% [markdown]
# ## Data Quality Assessment - Content Format Analysis
# 
# Initial assessment: count entries with raw HTML or JSON wrappers. This analysis reveals:
# - **HTML entries**: Articles wrapped in HTML tags that need conversion to plain text
# - **JSON entries**: Articles wrapped in JSON format with "articleBody" fields that need extraction
# 
# These counts help us understand the scope of preprocessing required and ensure we don't lose any data during cleaning.

# %%
num_html = df['text'].str.startswith('<html').sum()
num_json = df['text'].str.startswith('"articleBody":"').sum()
print(f"Rows starting with HTML: {num_html}")
print(f"Rows starting with JSON articleBody: {num_json}")

# %% [markdown]
# ## Entity Format Standardization
# 
# Prepare data for fine-tuning by normalizing entity formats:
# - **Fill missing values**: Ensures consistency in text processing by replacing NaN values with empty strings
# - **Extract entity names**: Remove ID metadata from entity fields, keeping only the actual names
# - **Standardize delimiters**: Convert all entity lists to semicolon-separated format for consistency
# 
# This step focuses the model training on raw text rather than metadata, improving generalization.

# %%
df_prep = df.copy().fillna('')
df_prep['persons'] = (
    df_prep['persons']
    .apply(lambda x: ';'.join([e.split(',')[0] for e in x.split(';')]))
)
df_prep['organizations'] = (
    df_prep['organizations']
    .apply(lambda x: ';'.join([e.split(',')[0] for e in x.split(';')]))
)
df_prep['locations'] = (
    df_prep['locations']
    .apply(lambda x: ';'.join(x.split(',')))
)
df_prep.head()

# %% [markdown]
# ## Text Content Cleaning and Normalization
# 
# Clean text column by addressing different content formats:
# - **HTML conversion**: Convert HTML segments to Markdown format, preserving readability while removing markup tags
# - **JSON extraction**: Extract plain text from JSON-wrapped "articleBody" fields, removing JSON structure
# - **Tag removal**: Strip specific HTML tags like `<a>` links that don't contribute to content understanding
# 
# This ensures all text is in a consistent, clean format suitable for NLP model training.

# %%
df_clean = df_prep.copy()
# Convert HTML segments to Markdown, removing <a> tags
html_idx = df_clean['text'].str.startswith('<html')
df_clean.loc[html_idx, 'text'] = (
    df_clean.loc[html_idx, 'text']
    .apply(lambda x: md(x, strip=['a']))
)
# Extract plain text from JSON-wrapped fields
json_idx = df_clean['text'].str.startswith('"articleBody":"')
df_clean.loc[json_idx, 'text'] = (
    df_clean.loc[json_idx, 'text']
    .apply(lambda x: x.split('"articleBody":"')[1].split('"')[0])
)
df_clean.head(15)

# %% [markdown]
# ## Cleaning Validation
# 
# Re-count entries after cleaning to verify that our transformations were successful:
# - **HTML remaining**: Should be 0 if all HTML was properly converted
# - **JSON remaining**: Should be 0 if all JSON wrappers were properly extracted
# 
# This validation step ensures no data was lost or improperly processed during cleaning.

# %%
pct_html_remaining = df_clean['text'].str.startswith('<html').sum()
pct_json_remaining = df_clean['text'].str.startswith('"articleBody":"').sum()
print(f"HTML entries remaining: {pct_html_remaining}")
print(f"JSON wrappers remaining: {pct_json_remaining}")

# %% [markdown]
# ## Dataset Splitting for Training and Evaluation
# 
# Split the cleaned dataset into training and testing sets to enable proper model evaluation:
# - **50% split**: Ensures balanced datasets for both training and evaluation
# - **Fixed random seed**: Guarantees reproducible splits across different runs
# - **Stratified approach**: Maintains distribution of entity types across splits
# 
# This separation is crucial for preventing data leakage and ensuring unbiased model performance assessment.

# %%
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df_clean,
    test_size=0.5,
    random_state=42
)
print(f"Training size: {len(train_df)}, Testing size: {len(test_df)}")

# %% [markdown]
# ## Token Distribution Analysis
# 
# Analyze token distribution for training data using tiktoken to understand text length characteristics:
# - **Model-specific tokenization**: Uses GPT-4 tokenizer for accurate token counting
# - **Length distribution**: Helps set appropriate sequence length limits for model training
# - **Memory planning**: Informs batch size and model architecture decisions
# - **Outlier detection**: Identifies extremely long or short texts that might need special handling

# %%
import tiktoken
enc = tiktoken.encoding_for_model('gpt-4o')
train_df['num_tokens'] = train_df['text'].apply(lambda x: len(enc.encode(x)))
train_df['num_tokens'].hist()
plt.title('Token Count Distribution')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ## Export Cleaned Datasets
# 
# Save the cleaned and split datasets for use in model fine-tuning:
# - **Training dataset**: Contains the data used for model training with token counts
# - **Testing dataset**: Reserved for final model evaluation and performance assessment
# 
# These files serve as the foundation for all subsequent model development and evaluation activities.

# %%
train_df.to_csv(DATASETS_DIR / 'full_data_clean_finetune.csv', index=False)
test_df.to_csv(DATASETS_DIR / 'full_data_clean.csv', index=False)

# %% [markdown]
# ## Optional: Text File Export for Batch Processing
# 
# (Optional) Export training text lines to a text file for batch fine-tuning scenarios where raw text input is preferred over structured CSV format. This can be useful for certain fine-tuning frameworks or when working with very large datasets.

# %%
# with open(FILES_DIR / 'finetune_data.txt', 'w', encoding='utf-8') as f:
#     for line in train_df['text']:
#         f.write(line + '\n')
