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

from notebooks import FILES_DIR

# Setup


encoding = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


df = pd.read_csv(FILES_DIR / "full_data_clean_finetune.csv")
articles = df["text"].tolist()

# %%
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    model_name="gpt-4o", chunk_size=500, chunk_overlap=0
)

texts = text_splitter.split_text(articles[0])
# %%
print(texts[0], file=open('test.txt', 'w'))
# %%
