# %% [markdown]
# # Data Exploration and Initial Analysis
# 
# This notebook performs comprehensive exploration of the raw dataset to understand its structure, content, and characteristics. We'll examine the data format, entity relationships, and identify any data quality issues that need to be addressed before model training.
# 
# ## Setup: initialize environment and import required libraries

# %%
from notebook_config import DATASETS_DIR, FILES_DIR
import pandas as pd
import json

# %% [markdown]
# ## Load and Examine Raw Dataset
# 
# Load the full dataset into a DataFrame to begin our exploration. This step establishes our baseline understanding of the data volume and structure.

# %%
df = pd.read_csv(DATASETS_DIR / 'full_data.csv')

# %% [markdown]
# ## Initial Data Preview
# 
# Display the first five rows of the DataFrame to get a quick overview of the data structure, column names, and typical content patterns. This helps identify the format of text fields and entity annotations.

# %%
df.head()

# %% [markdown]
# ## End of Dataset Preview
# 
# Display the last five rows to check for any patterns or differences at the end of the dataset, ensuring our data is consistent throughout.

# %%
df.tail()

# %% [markdown]
# ## Dataset Structure Analysis
# 
# Show summary of DataFrame structure and data types. This reveals:
# - Total number of rows and columns
# - Memory usage
# - Data types for each column (helpful for identifying string vs numeric fields)
# - Presence of missing values (NaN counts)

# %%
df.info()

# %% [markdown]
# ## Statistical Summary
# 
# Compute basic statistical summary for numeric columns. This provides insights into:
# - Central tendency measures (mean, median)
# - Dispersion measures (std, min, max, quartiles)
# - Distribution characteristics for any numerical features

# %%
df.describe()

# %% [markdown]
# ## Column Inventory
# 
# List all column names in the DataFrame to understand the complete feature set available for analysis and model training.

# %%
df.columns

# %% [markdown]
# ## Detailed Row Inspection - First Sample
# 
# Inspect all values in the first row to understand the complete structure of a single data point, including how entities are formatted and what metadata is available.

# %%
df.iloc[0]

# %% [markdown]
# ## Text Content Analysis - First Sample
# 
# View the text field of the first row to understand the content format, length, and style. This helps determine if text preprocessing will be needed (HTML removal, JSON parsing, etc.).

# %%
df.iloc[0]['text']

# %% [markdown]
# ## Entity Analysis - Person Entities
# 
# Examine person entities and their IDs in the first row to understand:
# - How person names are formatted
# - The relationship between names and their unique identifiers
# - Whether entities are comma-separated or use other delimiters

# %%
df.iloc[0]['persons']

# %% [markdown]
# ## Entity Analysis - Organization Entities
# 
# Examine organization entities and their IDs in the first row to understand the same formatting patterns for organizational entities, which may differ from person entities.

# %%
df.iloc[0]['organizations']

# %% [markdown]
# ## Entity Analysis - Theme Labels
# 
# View theme labels in the first row to understand how thematic categorization is implemented and what types of themes are present in the dataset.

# %%
df.iloc[0]['themes']

# %% [markdown]
# ## Entity Analysis - Location Entities
# 
# Inspect location labels in the first row to understand how geographical entities are annotated and whether they follow the same format as other entity types.

# %%
df.iloc[0]['locations']

# %% [markdown]
# ## Detailed Row Inspection - Second Sample
# 
# Inspect all values in the 11th row to compare with the first sample and identify any variations in data format or content patterns across different articles.

# %%
df.iloc[10]

# %% [markdown]
# ## Text Content Analysis - Second Sample
# 
# View the text field of the 11th row to compare content format and identify any differences in text structure, length, or formatting that might require different preprocessing approaches.

# %%
df.iloc[10]['text']

# %% [markdown]
# ## Data Quality Assessment - JSON Wrapper Detection
# 
# Count rows where text starts with "articleBody" to identify articles that are wrapped in JSON format. This helps determine the scope of JSON parsing needed during data preprocessing.

# %%
df[df['text'].str.startswith('"articleBody"')].shape[0]

# %% [markdown]
# ## Data Quality Assessment - HTML Content Detection
# 
# Count rows where text starts with an HTML tag to identify articles containing HTML markup. This helps determine the scope of HTML cleaning needed during preprocessing.

# %%
df[df['text'].str.startswith('<html')].shape[0]

# %% [markdown]
# ## Entity Mapping Construction
# 
# Build dictionary mapping entity IDs to lists of names for persons, organizations, and locations. This creates a comprehensive lookup table that:
# - Maps each unique entity ID to all its name variations
# - Helps identify entity disambiguation patterns
# - Provides insights into entity frequency and distribution
# - Enables reverse lookup from names to IDs for validation

# %%
entities_dict = {
    'persons': {},
    'organizations': {},
    'locations': {}
}
for i in range(len(df)):
    persons = df.iloc[i]['persons']
    organizations = df.iloc[i]['organizations']
    locations = df.iloc[i]['locations']

    if not pd.isna(persons):
        for person in persons.split(';'):
            p, _id = person.split(',')
            entities_dict['persons'].setdefault(_id, []).append(p)
    if not pd.isna(organizations):
        for organization in organizations.split(';'):
            o, _id = organization.split(',')
            entities_dict['organizations'].setdefault(_id, []).append(o)
    if not pd.isna(locations):
        for location in locations.split(','):
            l, _id = location, str(hash(location))
            entities_dict['locations'].setdefault(_id, []).append(l)

print(json.dumps(entities_dict, indent=4))
json.dump(entities_dict, open(FILES_DIR / 'misc' / 'entities_dict.json', 'w'), indent=4)

# %% [markdown]
# ## Reverse Entity Mapping Construction
# 
# Build reverse mapping from entity names to their corresponding IDs. This creates a complementary lookup that:
# - Enables finding all IDs associated with a given entity name
# - Helps identify potential entity linking issues
# - Supports entity normalization and deduplication
# - Provides validation capabilities for entity extraction models

# %%
entities_reversed_dict = {
    'persons': {},
    'organizations': {},
    'locations': {}
}
for _type, elements in entities_dict.items():
    for k, list_of_entities in elements.items():
        for name in list_of_entities:
            entities_reversed_dict[_type].setdefault(name, []).append(k)

print(json.dumps(entities_reversed_dict, indent=4))
json.dump(entities_reversed_dict, open(FILES_DIR / 'misc' / 'entities_reversed_dict.json', 'w'), indent=4)

# %%
# End of interactive script
