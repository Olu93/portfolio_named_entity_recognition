# %% 
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from notebooks import FILES_DIR
from src.port.entity_extractor import MultiEntityExtractor
from src.adapter.ootb.huggingface import HuggingFaceEntityExtractor
from src.adapter.naive.sliding_window import SlidingWindowExtractor
from sklearn.model_selection import train_test_split
# %%

df = pd.read_csv(FILES_DIR / 'full_data_clean.csv').fillna('')
# %%
df.head()


# %%
# Split data into train and test
train_df, test_df = train_test_split(df, test_size=0.6, random_state=42)
train_df.head()
# %%
extractor = MultiEntityExtractor()
extractor.add_extractor("persons", SlidingWindowExtractor())
extractor.add_extractor("organizations", SlidingWindowExtractor())
extractor.add_extractor("locations", SlidingWindowExtractor())
extractor.fit(train_df['text'], train_df[['persons', 'organizations', 'locations']])
extractor.predict(test_df['text'])
# %%
extractor_hf = MultiEntityExtractor()
extractor_hf.add_extractor("persons", HuggingFaceEntityExtractor(model="dslim/bert-base-NER", labels=["PER"]))
extractor_hf.add_extractor("organizations", HuggingFaceEntityExtractor(model="dslim/bert-base-NER", labels=["ORG"]))
extractor_hf.add_extractor("locations", HuggingFaceEntityExtractor(model="dslim/bert-base-NER", labels=["LOC"]))
extractor_hf.fit(train_df['text'], train_df[['persons', 'organizations', 'locations']])
extractor_hf.predict(test_df['text'])
# %%
test_df[['persons', 'organizations', 'locations']].head()