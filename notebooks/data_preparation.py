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
from markdownify import markdownify as md
# %%
df = pd.read_csv(FILES_DIR / 'full_data.csv')
df
# %%
df_prep = df.copy().fillna('')
df_prep["persons"] = df_prep["persons"].apply(lambda x: ";".join([e.split(',')[0] for e in x.split(';')]))
df_prep["organizations"] = df_prep["organizations"].apply(lambda x: ";".join([e.split(',')[0] for e in x.split(';')]))
df_prep["locations"] = df_prep["locations"].apply(lambda x: ";".join(x.split(',')))

df_prep
# %%
df_prep_clean = df_prep.copy()
html_indices = df_prep_clean["text"].str.startswith("<html")
df_prep_clean.loc[html_indices, 'text'] = df_prep_clean.loc[html_indices, 'text'].apply(lambda x: md(x))
df_prep_clean.head(15)
# %%


# %%
df_prep_clean.to_csv(FILES_DIR / 'full_data_clean.csv', index=False)
# %%