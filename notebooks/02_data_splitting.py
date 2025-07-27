# %%
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken

from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter

from notebooks import FILES_DIR

# Setup


encoding = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


df = pd.read_csv(FILES_DIR / "full_data_clean_finetune.csv")
articles = df["text"].tolist()

# %%
# === PASS 1: Evaluate multiple configs on first 5 articles ===

threshold_configs = [
    ("percentile", 80.0),
    ("percentile", 90.0),
    ("percentile", 95.0),
    ("percentile", 98.0),
    ("standard_deviation", 1.5),
    ("interquartile", 1.5)
]

stats_data, results_json = [], []

for threshold_type, threshold_amount in tqdm(threshold_configs, desc="Configs"):
    splitter = SemanticChunker(
        OpenAIEmbeddings(),
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )

    for i, article in enumerate(articles[:5]):
        docs = splitter.create_documents([article])
        tokens = [count_tokens(doc.page_content) for doc in docs]

        stats_data.append(
            {
                "config": f"{threshold_type} {threshold_amount}",
                "article_index": i,
                "num_chunks": len(docs),
                "min_tokens": min(tokens),
                "max_tokens": max(tokens),
                "mean_tokens": sum(tokens) / len(tokens),
                "median_tokens": sorted(tokens)[len(tokens) // 2],
            }
        )

        results_json.append(
            {
                "config": f"{threshold_type} {threshold_amount}",
                "article_index": i,
                "original_length": len(article),
                "original_tokens": count_tokens(article),
                "chunks": [
                    {
                        "chunk_index": j,
                        "length": len(doc.page_content),
                        "tokens": count_tokens(doc.page_content),
                        "content": doc.page_content,
                    }
                    for j, doc in enumerate(docs)
                ],
            }
        )

# Save evaluation results
eval_stats_path = FILES_DIR / "semantic_eval_stats_first5.csv"
eval_json_path = FILES_DIR / "semantic_eval_chunks_first5.json"
pd.DataFrame(stats_data).to_csv(eval_stats_path, index=False)
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

# %%
# === PLOTTING ===
stats_df = pd.DataFrame(stats_data)

plt.style.use("default")
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "Semantic Chunking Config Comparison (First 5 Articles)",
    fontsize=16,
    fontweight="bold",
)

sns.barplot(data=stats_df, x="config", y="median_tokens", ax=axes[0, 0])
axes[0, 0].set_title("Median Tokens per Chunk")
axes[0, 0].tick_params(axis="x", rotation=45)

sns.barplot(data=stats_df, x="config", y="mean_tokens", ax=axes[0, 1])
axes[0, 1].set_title("Mean Tokens per Chunk")
axes[0, 1].tick_params(axis="x", rotation=45)

sns.barplot(data=stats_df, x="config", y="num_chunks", ax=axes[1, 0])
axes[1, 0].set_title("Chunks per Article")
axes[1, 0].tick_params(axis="x", rotation=45)

sns.boxplot(data=stats_df, x="config", y="num_chunks", ax=axes[1, 1])
axes[1, 1].set_title("Chunk Distribution")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plot_path = FILES_DIR / "semantic_chunking_eval_first5_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved stats CSV: {eval_stats_path}")
print(f"Saved JSON output: {eval_json_path}")
print(f"Saved plot: {plot_path}")

# %%
# === PASS 2: Process full corpus with best config ===

best_type, best_amount = "percentile", 80.0
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type=best_type,
    breakpoint_threshold_amount=best_amount,
)

text_splitter = TokenTextSplitter(
    model_name="gpt-4o", chunk_size=500, chunk_overlap=0
)

all_chunks = []
for i, article in enumerate(tqdm(articles, desc="Processing full corpus")):
    docs = splitter.create_documents([article])
    chunks = text_splitter.split_documents(docs)

    orig_row = df.iloc[i].to_dict()

    for j, chunk in enumerate(chunks):
        new_row = orig_row.copy()
        new_row.update(
            {
                "chunk_id": f"{i}_{j}",
                "original_article_index": i,
                "chunk_index": j,
                "text": chunk.page_content,
                "num_tokens": count_tokens(chunk.page_content),
                "num_chunks_in_original": len(docs),
                "threshold_type": best_type,
                "threshold_amount": best_amount,
            }
        )
        all_chunks.append(new_row)

    output_df = pd.DataFrame(all_chunks)
    final_csv = FILES_DIR / "semantic_split_complete_dataset.csv"
    output_df.to_csv(final_csv, index=False)

print(f"\n✅ Saved full processed dataset: {final_csv}")
print(f"Total chunks: {len(output_df)}")
print(f"Chunks per article (mean): {len(output_df)/len(articles):.2f}")
print(
    f"Token stats — Min: {output_df.num_tokens.min()}, Max: {output_df.num_tokens.max()}, Mean: {output_df.num_tokens.mean():.1f}"
)

# %%
