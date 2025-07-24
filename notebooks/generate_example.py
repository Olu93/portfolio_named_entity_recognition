# %% 
import sys
import pathlib

# sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'src'))
from utils.preprocessing import take_person_or_org
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from notebooks import FILES_DIR
# %%
df = pd.read_csv(FILES_DIR / 'full_data.csv')
df
# %%
new_df = df.dropna(subset=['persons', 'organizations', 'locations']).copy()
new_df["persons"] = new_df["persons"].apply(lambda x: x.split(';'))
new_df["organizations"] = new_df["organizations"].apply(lambda x: x.split(';'))
new_df["locations"] = new_df["locations"].apply(lambda x: x.split(','))
new_df
# %%#
final_df = new_df.copy()
final_df = final_df.drop(columns=['Unnamed: 0', 'themes'])
final_df['persons'] = final_df['persons'].apply(lambda x: [take_person_or_org(e) for e in x])
final_df['organizations'] = final_df['organizations'].apply(lambda x: [take_person_or_org(e) for e in x])
final_df['locations'] = final_df['locations'].apply(lambda x: [e for e in x])
final_df
# %%
final_df.to_dict(orient='records')
# %%
json.dump(final_df[:2].to_dict(orient='records'), open(FILES_DIR / 'misc/examples.json', 'w'), indent=4)
# %%
json.dump(final_df[:20].to_dict(orient='records'), open(FILES_DIR / 'misc/examples_many.json', 'w'), indent=4)
# %%