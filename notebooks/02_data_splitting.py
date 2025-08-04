# %% [markdown]
# # Advanced Data Splitting and Semantic Chunking
# 
#  This notebook implements sophisticated text chunking strategies to prepare the dataset for Named Entity Recognition (NER) model training. We'll evaluate different semantic chunking configurations and create optimally-sized text segments that preserve semantic coherence while fitting within model token limits.
# 
#  ## Setup environment and import required libraries for chunking and tokenization

# %%
# %%

from notebook_config import DATASETS_DIR, FILES_DIR, MISC_DIR
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken

from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter

# Initialize tokenizer for counting tokens
encoding = tiktoken.encoding_for_model("gpt-4o")
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# %% [markdown]
# ## Load Preprocessed Dataset
# 
#  Load the cleaned dataset prepared for fine-tuning from the previous notebook. This dataset contains:
#  - **Cleaned text**: HTML and JSON content has been normalized
#  - **Standardized entities**: Entity names without ID metadata
#  - **Token counts**: Pre-computed token lengths for each article
# 
#  This serves as our foundation for implementing advanced chunking strategies.

# %%
# %%

df = pd.read_csv(DATASETS_DIR / "full_data_clean_finetune.csv")
articles = df["text"].tolist()

# %% [markdown]
# ## Phase 1: Semantic Chunking Configuration Evaluation
# 
#  Evaluate different semantic chunking configurations on a small sample to identify optimal parameters:
# 
#  **Threshold Strategies Tested:**
#  - **Percentile-based**: Uses statistical percentiles (80%, 90%, 95%, 98%) to determine breakpoints
#  - **Standard deviation**: Uses mean ± 1.5σ to identify natural text boundaries
#  - **Interquartile range**: Uses IQR-based thresholds for robust breakpoint detection
# 
#  **Evaluation Metrics:**
#  - Chunk count per article
#  - Token distribution statistics (min, max, mean, median)
#  - Semantic coherence preservation
# 
#  This systematic evaluation ensures we choose the most effective chunking strategy for our specific dataset.

# %%
# %%

import spacy
from langchain.text_splitter import TextSplitter
from typing import List

class SpacySentenceSplitter(TextSplitter):
    def __init__(
        self,
        model: str = "en_core_web_sm",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        length_function=None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                         length_function=length_function or len)
        self.nlp = spacy.load(model, disable=["ner", "tagger", "parser"])
        self.nlp.add_pipe('sentencizer')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences



splitter = SpacySentenceSplitter()

stats_data = []
results_json = []

for i, article in enumerate(articles[:10]):
    sentences = splitter.create_documents([article])
    # tokens = [count_tokens(doc.page_content) for doc in docs]
    # Collect summary statistics for each chunking config
    for sent in sentences:
        stats_data.append({
            "article_index": i,
            "num_tokens": len(encoding.encode(sent.page_content)),
    })
    # Store full chunk details for inspection
    results_json.append({
        "article_index": i,
        "original_tokens": len(encoding.encode(article)),
        "chunks": [
            {
                "chunk_index": j,
                "chars": len(sent.page_content),
                "tokens": len(encoding.encode(sent.page_content)),
                "content": sent.page_content,
            }
            for j, sent in enumerate(sentences)
        ],
    })

# Save evaluation outputs for later review
# - CSV of summary stats
# - JSON with detailed chunk content
eval_stats_path = MISC_DIR / "semantic_eval_stats_first5.csv"
eval_json_path = MISC_DIR / "semantic_eval_chunks_first5.json"
pd.DataFrame(stats_data).to_csv(eval_stats_path, index=False)
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"Saved stats CSV: {eval_stats_path}")
print(f"Saved JSON output: {eval_json_path}")

# %% [markdown]
# ## Configuration Comparison Visualization
# 
#  Create comprehensive visualizations to compare chunking configurations across multiple dimensions:
# 
#  **Metrics Visualized:**
#  - **Median tokens per chunk**: Central tendency of chunk sizes
#  - **Mean tokens per chunk**: Average chunk size with outlier sensitivity
#  - **Chunks per article**: Granularity of text segmentation
#  - **Chunk distribution**: Spread and variability of chunk sizes
# 
#  These visualizations help identify the optimal balance between semantic coherence and practical token limits for model training.

# %%
# %%

stats_df = pd.DataFrame(stats_data)
plt.style.use("default")
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "Semantic Chunking Config Comparison (First 5 Articles)",
    fontsize=16,
    fontweight="bold",
)

sns.barplot(data=stats_df.groupby("article_index").median(), x="article_index", y="num_tokens", ax=axes[0, 0])
axes[0, 0].set_title("Median Tokens per Chunk")
axes[0, 0].tick_params(axis="x")

sns.barplot(data=stats_df.groupby("article_index").mean(), x="article_index", y="num_tokens", ax=axes[0, 1])
axes[0, 1].set_title("Mean Tokens per Chunk")
axes[0, 1].tick_params(axis="x")

sns.barplot(data=stats_df.groupby("article_index").count(), x="article_index", y="num_tokens", ax=axes[1, 0])
axes[1, 0].set_title("Chunks per Article")
axes[1, 0].tick_params(axis="x")

sns.boxplot(data=stats_df, x="article_index", y="num_tokens", ax=axes[1, 1])
axes[1, 1].set_title("Chunk Distribution")
axes[1, 1].tick_params(axis="x")

plt.tight_layout()
plot_path = MISC_DIR / "semantic_chunking_eval_first5_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved plot: {plot_path}")

# %% [markdown]
# ## Phase 2: Full Corpus Processing with Optimal Configuration
# 
#  Apply the best-performing chunking configuration to the entire dataset:
# 
#  **Two-Stage Processing:**
#  1. **Semantic chunking**: Uses percentile-based threshold (80%) to create semantically coherent segments
#  2. **Token-based splitting**: Further divides semantic chunks into fixed 500-token segments for NER labeling
# 
#  **Benefits of This Approach:**
#  - Preserves semantic boundaries and context
#  - Ensures consistent token lengths for model training
#  - Maintains entity relationships within chunks
#  - Optimizes memory usage and batch processing efficiency

# %%
# %%

best_type, best_amount = "percentile", 80.0
splitter = SpacySentenceSplitter()
text_splitter = TokenTextSplitter(
    model_name="gpt-4o", chunk_size=500, chunk_overlap=0
)

all_chunks = []
for i, article in enumerate(tqdm(articles, desc="Processing full corpus")):
    docs = splitter.create_documents([article])
    chunks = text_splitter.split_documents(docs)
    orig_row = df.iloc[i].to_dict()
    for j, chunk in enumerate(chunks):
        row = orig_row.copy()
        row.update({
            "chunk_id": f"{i}_{j}",
            "original_article_index": i,
            "chunk_index": j,
            "text": chunk.page_content,
            "num_tokens": count_tokens(chunk.page_content),
            "num_chunks_in_original": len(docs),
        })
        all_chunks.append(row)
# Save the complete chunked dataset
output_df = pd.DataFrame(all_chunks)
final_csv = DATASETS_DIR / "semantic_split_complete_dataset.csv"
output_df.to_csv(final_csv, index=False)

print(f"✅ Saved full processed dataset: {final_csv}")
print(f"Total chunks: {len(output_df)}")
print(f"Chunks per article (mean): {len(output_df)/len(articles):.2f}")
print(
    f"Token stats — Min: {output_df.num_tokens.min()}, Max: {output_df.num_tokens.max()}, Mean: {output_df.num_tokens.mean():.1f}"
)

# %% [markdown]
# ## Final Token Distribution Analysis
# 
#  Visualize the token distribution in the final chunked dataset to validate our chunking strategy:
# 
#  **Key Insights:**
#  - **Distribution shape**: Confirms chunks are appropriately sized for model training
#  - **Outlier detection**: Identifies any chunks that are too long or too short
#  - **Model compatibility**: Ensures chunks fit within typical transformer model sequence limits
#  - **Training efficiency**: Validates that chunk sizes support effective batch processing
# 
#  This analysis ensures our chunked dataset is optimally prepared for NER model training and evaluation.

# %%
# %%

output_df.num_tokens.hist()
plt.title('Final Chunk Token Distribution')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.show()



# %%
